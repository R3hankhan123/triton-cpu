#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Rank >= 2 elementwise ops become a loop over the leading dimension.
// Unrolling that dimension outright is what makes vLLM prefill kernels take
// minutes to compile on VXE.
struct PeelLeadingVectorDim : public RewritePattern {
  PeelLeadingVectorDim(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/2, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();
    if (isa<arith::ConstantOp>(op))
      return failure();

    auto vecTy = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!vecTy || vecTy.getRank() < 2 || !vecTy.hasStaticShape())
      return failure();

    for (Value operand : op->getOperands()) {
      auto operandTy = dyn_cast<VectorType>(operand.getType());
      if (operandTy && operandTy.getShape() != vecTy.getShape())
        return failure();
    }

    Location loc = op->getLoc();
    auto rowShape = vecTy.getShape().drop_front();
    auto rowTy = VectorType::get(rowShape, vecTy.getElementType());
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(vecTy));
    Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value ub =
        arith::ConstantIndexOp::create(rewriter, loc, vecTy.getDimSize(0));
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto loop = scf::ForOp::create(
        rewriter, loc, lb, ub, step, ValueRange{zero},
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
          SmallVector<Value> operands;
          operands.reserve(op->getNumOperands());
          for (Value operand : op->getOperands()) {
            if (isa<VectorType>(operand.getType())) {
              operands.push_back(vector::ExtractOp::create(b, bodyLoc, operand,
                                                           OpFoldResult(iv)));
            } else {
              operands.push_back(operand);
            }
          }
          OperationState state(bodyLoc, op->getName());
          state.addOperands(operands);
          state.addTypes(rowTy);
          state.addAttributes(op->getAttrs());
          Operation *inner = b.create(state);
          Value inserted = vector::InsertOp::create(
              b, bodyLoc, inner->getResult(0), iterArgs[0], OpFoldResult(iv));
          scf::YieldOp::create(b, bodyLoc, inserted);
        });
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

VectorType vectorOperandType(Operation *op) {
  if (auto write = dyn_cast<vector::TransferWriteOp>(op))
    return write.getVectorType();
  if (auto store = dyn_cast<vector::StoreOp>(op))
    return dyn_cast<VectorType>(store.getValueToStore().getType());
  if (op->getNumResults() == 1)
    return dyn_cast<VectorType>(op->getResult(0).getType());
  return nullptr;
}

struct SplitWideVectorsPass
    : public PassWrapper<SplitWideVectorsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitWideVectorsPass);

  unsigned nativeBitWidth = 128;

  SplitWideVectorsPass() = default;
  explicit SplitWideVectorsPass(unsigned nativeBitWidth)
      : nativeBitWidth(nativeBitWidth) {}

  StringRef getArgument() const final {
    return "triton-cpu-split-wide-vectors";
  }
  StringRef getDescription() const final {
    return "Lower wide vectors to native VXE registers";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    unsigned bits = nativeBitWidth;
    vector::UnrollVectorOptions unrollOptions;
    unrollOptions.setNativeShapeFn(
        [bits](Operation *op) -> std::optional<SmallVector<int64_t>> {
          VectorType vecTy = vectorOperandType(op);
          if (!vecTy || vecTy.getRank() != 1 || !vecTy.hasStaticShape())
            return std::nullopt;
          unsigned elemBits = vecTy.getElementTypeBitWidth();
          if (elemBits == 0 || bits % elemBits != 0)
            return std::nullopt;
          int64_t nativeElems = bits / elemBits;
          int64_t width = vecTy.getShape().back();
          if (nativeElems <= 0 || width <= nativeElems ||
              width % nativeElems != 0)
            return std::nullopt;
          return SmallVector<int64_t>{nativeElems};
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<PeelLeadingVectorDim>(&getContext());
    vector::populateVectorUnrollPatterns(patterns, unrollOptions);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::cpu::createSplitWideVectors(unsigned nativeVectorBitWidth) {
  return std::make_unique<SplitWideVectorsPass>(nativeVectorBitWidth);
}
