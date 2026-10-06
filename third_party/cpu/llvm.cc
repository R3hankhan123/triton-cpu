#include "triton/Tools/Sys/GetEnv.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

namespace py = nanobind;

namespace {

std::string getHostTargetTriple() {
  std::string triple = llvm::sys::getDefaultTargetTriple();
  if (triple.empty())
    triple = llvm::sys::getProcessTriple();
  return triple;
}

void initializeHostTarget() {
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    if (llvm::InitializeNativeTarget())
      throw std::runtime_error("LLVM native target is not available");
    if (llvm::InitializeNativeTargetAsmPrinter())
      throw std::runtime_error("LLVM native target assembly printer is not "
                               "available");
    // SystemZ schedules legacy codegen passes (EarlyIfConverter, IfConverter,
    // MachineCombiner) that InitializeNativeTarget does not register. With
    // assertions enabled, the legacy pass manager aborts if they are missing.
    llvm::PassRegistry &registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(registry);
    llvm::initializeCodeGen(registry);
    llvm::initializeLoopStrengthReducePass(registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(registry);
    llvm::initializeConstantHoistingLegacyPassPass(registry);
    llvm::initializeScalarOpts(registry);
    llvm::initializeIPO(registry);
    llvm::initializeVectorization(registry);
    llvm::initializeScalarizeMaskedMemIntrinLegacyPassPass(registry);
    llvm::initializeTransformUtils(registry);
  });

  // LLVM's global thread pool is not fork-safe, so other targets stay
  // single-threaded. s390x vLLM kernels are large enough that a small pool
  // pays off, and it starts at compile time inside the already-spawned worker.
  unsigned threads = 1;
  if (llvm::Triple(getHostTargetTriple()).isSystemZ())
    threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  if (const char *override = std::getenv("TRITON_CPU_LLVM_THREADS")) {
    unsigned parsed = std::atoi(override);
    if (parsed > 0)
      threads = parsed;
  }
  llvm::parallel::strategy = llvm::hardware_concurrency(threads);
}

void setLLVMBooleanOption(const std::string &name, bool value) {
  auto options = llvm::cl::getRegisteredOptions();
  auto it = options.find(name);
  if (it == options.end())
    return;
  it->second->addOccurrence(1, name, value ? "true" : "false");
}

std::unique_ptr<llvm::TargetMachine>
createHostTargetMachine(llvm::Module &module, bool enableFpFusion,
                        bool enableFastMath) {
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  if (!target)
    throw std::runtime_error("target lookup error: " + error);

  llvm::TargetOptions options;
  if (enableFpFusion)
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  if (enableFastMath)
    options.NoTrappingFPMath = true;
  options.TrapUnreachable = true;
  // Verbose assembly comments dominate compile time on the large kernels
  // vLLM emits. The integrated assembler does not need them.
  bool verboseAsm = !llvm::Triple(module.getTargetTriple()).isSystemZ();
  options.MCOptions.AsmVerbose = verboseAsm;
  options.MCOptions.PreserveAsmComments = verboseAsm;

  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  return std::unique_ptr<llvm::TargetMachine>{target->createTargetMachine(
      module.getTargetTriple(), llvm::sys::getHostCPUName(), "", options,
      llvm::Reloc::PIC_, std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
}

std::string translateHostLLVMIRToASM(llvm::Module &module, bool enableFpFusion,
                                     bool enableFastMath) {
  if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP"))
    setLLVMBooleanOption("print-after-all", true);

  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<llvm::StringRef, 3> flags;
      llvm::StringRef(flagList).split(flags, ',');
      for (llvm::StringRef flag : flags)
        setLLVMBooleanOption(flag.str(), true);
    }
  }

  for (llvm::Function &function : module.functions())
    if (!function.hasFnAttribute(llvm::Attribute::NoInline))
      function.addFnAttr(llvm::Attribute::AlwaysInline);

  llvm::legacy::PassManager modulePasses;
  modulePasses.add(llvm::createAlwaysInlinerLegacyPass());
  modulePasses.add(llvm::createVerifierPass());

  const bool enableTiming =
      mlir::triton::tools::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enableTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  modulePasses.run(module);

  llvm::SmallString<0> timePassesStr;
  llvm::raw_svector_ostream reportStream(timePassesStr);
  if (enableTiming) {
    llvm::reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }

  module.setTargetTriple(llvm::Triple(getHostTargetTriple()));
  auto machine =
      createHostTargetMachine(module, enableFpFusion, enableFastMath);
  module.setDataLayout(machine->createDataLayout());

  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream bufferedStream(stream);
    llvm::legacy::PassManager codegenPasses;
    machine->addPassesToEmitFile(codegenPasses, bufferedStream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile);
    codegenPasses.run(module);

    if (enableTiming) {
      llvm::reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
  }
  return result;
}

void setHostTarget(llvm::Module &module) {
  initializeHostTarget();
  module.setTargetTriple(llvm::Triple(getHostTargetTriple()));

  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  if (!target)
    throw std::runtime_error("target lookup error: " + error);

  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module.getTargetTriple(), llvm::sys::getHostCPUName(), "", {},
      llvm::Reloc::PIC_)};
  module.setDataLayout(machine->createDataLayout());
}

std::set<std::string> getCPUFeatures() {
  auto features = llvm::sys::getHostCPUFeatures();

  std::set<std::string> result;
  for (const auto &feature : features)
    if (feature.second)
      result.insert(feature.first().str());

  // NEON is mandatory on AArch64, and VXE is mandatory on the s390x hosts
  // this backend is built for. LLVM feature detection can return an empty
  // set for both; use the ISA name in that case.
  if (result.empty()) {
    std::string processTriple = llvm::sys::getProcessTriple();
    std::size_t separator = processTriple.find('-');
    if (separator != std::string::npos) {
      std::string arch = processTriple.substr(0, separator);
      if (arch == "aarch64" || arch == "arm64")
        result.insert("neon");
      else if (arch == "s390x")
        result.insert("vxe");
    }
  }

  return result;
}

} // namespace

void init_triton_cpu_llvm(py::module_ &m) {
  m.def("get_cpu_triple", []() { return llvm::sys::getProcessTriple(); });
  m.def("get_cpu_name", []() { return llvm::sys::getHostCPUName().str(); });
  m.def("get_cpu_features", &getCPUFeatures);
  m.def("set_host_target",
        [](llvm::Module *module) { setHostTarget(*module); });
  m.def("translate_to_asm",
        [](std::string llvmIR, bool enableFpFusion,
           bool enableFastMath) -> py::object {
          std::string result;
          {
            py::gil_scoped_release release;
            initializeHostTarget();

            llvm::LLVMContext context;
            std::unique_ptr<llvm::MemoryBuffer> buffer =
                llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
            llvm::SMDiagnostic error;
            std::unique_ptr<llvm::Module> module =
                llvm::parseIR(buffer->getMemBufferRef(), error, context);
            if (!module) {
              llvm::report_fatal_error(
                  "failed to parse IR: " + error.getMessage() +
                  "lineno: " + std::to_string(error.getLineNo()));
            }
            result = translateHostLLVMIRToASM(*module, enableFpFusion,
                                              enableFastMath);
          }
          return py::str(result.c_str(), result.size());
        });
}
