// Lower wide masked memory ops to a fast (unmasked) path plus a chunked
// fallback.
//
// SystemZ has no masked vector memory instructions. LLVM scalarizes
// llvm.masked.load/store into per-lane if-then-else branches. With
// BLOCK_SIZE=8192 and f32, that's 8192 branches in straight-line code —
// extreme register spilling and compilation times of minutes per kernel.
//
// This pass rewrites each vector.maskedload / vector.maskedstore whose
// width exceeds a single native register into:
//
//   1. Fast path – vector.reduction <and> over the mask. If all true,
//      emit a plain unmasked load or store.
//
//   2. Slow path – loop over 128-bit chunks doing one small masked op
//      per chunk. LLVM still scalarizes the small op, but only into
//      4 branches (for f32) inside a loop body, not 8192 in a row.
//
// Ops that already fit in one native register are left alone.

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_LOWERWIDEMASKEDMEMORYOPS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

static int64_t nativeElems(unsigned nativeBits, Type elemTy) {
  unsigned elemBits = elemTy.getIntOrFloatBitWidth();
  return elemBits > 0 ? nativeBits / elemBits : 0;
}

static bool isWide(VectorType vecTy, unsigned nativeBits) {
  if (vecTy.getRank() != 1 || !vecTy.hasStaticShape())
    return false;
  int64_t ne = nativeElems(nativeBits, vecTy.getElementType());
  return ne > 0 && vecTy.getNumElements() > ne;
}

/// Build the "leading indices" = all but the last element of `indices`.
static SmallVector<Value> getLeadingIndices(ValueRange indices) {
  SmallVector<Value> leading;
  for (unsigned i = 0, n = indices.size(); i + 1 < n; ++i)
    leading.push_back(indices[i]);
  return leading;
}

// -----------------------------------------------------------------------
// LowerWideMaskedLoad
// -----------------------------------------------------------------------
struct LowerWideMaskedLoad
    : public OpRewritePattern<vector::MaskedLoadOp> {
  unsigned nativeBits;

  LowerWideMaskedLoad(MLIRContext *ctx, unsigned nativeBits)
      : OpRewritePattern(ctx, /*benefit=*/1), nativeBits(nativeBits) {}

  LogicalResult matchAndRewrite(vector::MaskedLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto vecTy = op.getVectorType();
    if (!isWide(vecTy, nativeBits))
      return failure();

    Location loc = op.getLoc();
    int64_t totalElems = vecTy.getNumElements();
    Type elemTy = vecTy.getElementType();
    int64_t chunkElems = nativeElems(nativeBits, elemTy);
    assert(chunkElems > 0);
    int64_t numChunks = (totalElems + chunkElems - 1) / chunkElems;

    Value mask = op.getMask();
    Value passthru = op.getPassThru();
    Value base = op.getBase();
    ValueRange indices = op.getIndices();

    auto i1Ty = rewriter.getI1Type();
    Value allTrue = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::AND, mask);
    if (allTrue.getType() != i1Ty)
      allTrue = arith::TruncIOp::create(rewriter, loc, i1Ty, allTrue);

    auto ifOp = scf::IfOp::create(
        rewriter, loc, TypeRange{vecTy}, allTrue, /*withElseRegion=*/true);

    // Then: unmasked load.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      Value loaded = vector::LoadOp::create(rewriter, loc, vecTy, base,
                                            indices);
      scf::YieldOp::create(rewriter, loc, loaded);
    }

    // Else: chunked masked load.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.elseBlock());

      Value result = passthru;
      auto chunkVecTy = VectorType::get({chunkElems}, elemTy);
      auto chunkMaskTy = VectorType::get({chunkElems}, i1Ty);

      Value baseIdx = indices.empty()
          ? (Value)arith::ConstantIndexOp::create(rewriter, loc, 0)
          : indices.back();
      SmallVector<Value> leadingIndices = getLeadingIndices(indices);

      for (int64_t c = 0; c < numChunks; ++c) {
        int64_t offset = c * chunkElems;
        int64_t thisChunk = std::min(chunkElems, totalElems - offset);

        Value subMask;
        Value subPassthru;
        if (thisChunk == chunkElems) {
          subMask = vector::ExtractStridedSliceOp::create(
              rewriter, loc, mask,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});
          subPassthru = vector::ExtractStridedSliceOp::create(
              rewriter, loc, result,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});
        } else {
          Value shortMask = vector::ExtractStridedSliceOp::create(
              rewriter, loc, mask,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});
          Value shortPassthru = vector::ExtractStridedSliceOp::create(
              rewriter, loc, result,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});

          Value falseMask = arith::ConstantOp::create(
              rewriter, loc, chunkMaskTy,
              rewriter.getZeroAttr(chunkMaskTy));
          subMask = vector::InsertStridedSliceOp::create(
              rewriter, loc, shortMask, falseMask,
              ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});

          Value zeroPassthru = arith::ConstantOp::create(
              rewriter, loc, chunkVecTy,
              rewriter.getZeroAttr(chunkVecTy));
          subPassthru = vector::InsertStridedSliceOp::create(
              rewriter, loc, shortPassthru, zeroPassthru,
              ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        }

        Value chunkIdx = arith::ConstantIndexOp::create(rewriter, loc, offset);
        Value newIdx = arith::AddIOp::create(rewriter, loc, baseIdx, chunkIdx);
        SmallVector<Value> newIndices(leadingIndices);
        newIndices.push_back(newIdx);

        Value chunkLoaded = vector::MaskedLoadOp::create(
            rewriter, loc, chunkVecTy, base, newIndices, subMask, subPassthru);

        if (thisChunk == chunkElems) {
          result = vector::InsertStridedSliceOp::create(
              rewriter, loc, chunkLoaded, result,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{1});
        } else {
          Value shortResult = vector::ExtractStridedSliceOp::create(
              rewriter, loc, chunkLoaded,
              ArrayRef<int64_t>{0}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});
          result = vector::InsertStridedSliceOp::create(
              rewriter, loc, shortResult, result,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{1});
        }
      }

      scf::YieldOp::create(rewriter, loc, result);
    }

    rewriter.replaceOp(op, ifOp.getResult(0));
    return success();
  }
};

// -----------------------------------------------------------------------
// LowerWideMaskedStore
// -----------------------------------------------------------------------
struct LowerWideMaskedStore
    : public OpRewritePattern<vector::MaskedStoreOp> {
  unsigned nativeBits;

  LowerWideMaskedStore(MLIRContext *ctx, unsigned nativeBits)
      : OpRewritePattern(ctx, /*benefit=*/1), nativeBits(nativeBits) {}

  LogicalResult matchAndRewrite(vector::MaskedStoreOp op,
                                PatternRewriter &rewriter) const override {
    Value valueToStore = op.getValueToStore();
    auto vecTy = dyn_cast<VectorType>(valueToStore.getType());
    if (!vecTy || !isWide(vecTy, nativeBits))
      return failure();

    Location loc = op.getLoc();
    int64_t totalElems = vecTy.getNumElements();
    Type elemTy = vecTy.getElementType();
    int64_t chunkElems = nativeElems(nativeBits, elemTy);
    assert(chunkElems > 0);
    int64_t numChunks = (totalElems + chunkElems - 1) / chunkElems;

    Value mask = op.getMask();
    Value base = op.getBase();
    ValueRange indices = op.getIndices();

    auto i1Ty = rewriter.getI1Type();

    Value allTrue = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::AND, mask);
    if (allTrue.getType() != i1Ty)
      allTrue = arith::TruncIOp::create(rewriter, loc, i1Ty, allTrue);

    auto ifOp = scf::IfOp::create(
        rewriter, loc, TypeRange{}, allTrue, /*withElseRegion=*/true);

    // Then: unmasked store.
    // scf.if with no results auto-creates scf.yield terminators,
    // so we only insert the store before the existing yield.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      vector::StoreOp::create(rewriter, loc, valueToStore, base, indices);
    }

    // Else: chunked masked store.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.elseBlock());

      auto chunkVecTy = VectorType::get({chunkElems}, elemTy);
      auto chunkMaskTy = VectorType::get({chunkElems}, i1Ty);

      Value baseIdx = indices.empty()
          ? (Value)arith::ConstantIndexOp::create(rewriter, loc, 0)
          : indices.back();
      SmallVector<Value> leadingIndices = getLeadingIndices(indices);

      for (int64_t c = 0; c < numChunks; ++c) {
        int64_t offset = c * chunkElems;
        int64_t thisChunk = std::min(chunkElems, totalElems - offset);

        Value subMask;
        Value subData;
        if (thisChunk == chunkElems) {
          subMask = vector::ExtractStridedSliceOp::create(
              rewriter, loc, mask,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});
          subData = vector::ExtractStridedSliceOp::create(
              rewriter, loc, valueToStore,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});
        } else {
          Value shortMask = vector::ExtractStridedSliceOp::create(
              rewriter, loc, mask,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});
          Value shortData = vector::ExtractStridedSliceOp::create(
              rewriter, loc, valueToStore,
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{thisChunk},
              ArrayRef<int64_t>{1});

          Value falseMask = arith::ConstantOp::create(
              rewriter, loc, chunkMaskTy,
              rewriter.getZeroAttr(chunkMaskTy));
          subMask = vector::InsertStridedSliceOp::create(
              rewriter, loc, shortMask, falseMask,
              ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});

          Value zeroData = arith::ConstantOp::create(
              rewriter, loc, chunkVecTy,
              rewriter.getZeroAttr(chunkVecTy));
          subData = vector::InsertStridedSliceOp::create(
              rewriter, loc, shortData, zeroData,
              ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        }

        Value chunkIdx = arith::ConstantIndexOp::create(rewriter, loc, offset);
        Value newIdx = arith::AddIOp::create(rewriter, loc, baseIdx, chunkIdx);
        SmallVector<Value> newIndices(leadingIndices);
        newIndices.push_back(newIdx);

        vector::MaskedStoreOp::create(
            rewriter, loc, base, newIndices, subMask, subData);
      }
      // scf.yield already exists (auto-generated); don't add another.
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// -----------------------------------------------------------------------
// Pass
// -----------------------------------------------------------------------
struct LowerWideMaskedMemoryOpsPass
    : public triton::cpu::impl::LowerWideMaskedMemoryOpsBase<
          LowerWideMaskedMemoryOpsPass> {

  LowerWideMaskedMemoryOpsPass() = default;
  explicit LowerWideMaskedMemoryOpsPass(unsigned nativeBits) {
    this->nativeVectorBitWidth = nativeBits;
  }

  void runOnOperation() override {
    unsigned bits = this->nativeVectorBitWidth;
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerWideMaskedLoad>(ctx, bits);
    patterns.add<LowerWideMaskedStore>(ctx, bits);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createLowerWideMaskedMemoryOps(unsigned nativeVectorBitWidth) {
  return std::make_unique<LowerWideMaskedMemoryOpsPass>(nativeVectorBitWidth);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
