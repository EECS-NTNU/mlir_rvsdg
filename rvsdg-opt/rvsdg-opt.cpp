#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include <stdio.h>

#include <JLM/JLMDialect.h>
#include <RVSDG/RVSDGDialect.h>
#include <RVSDG/RVSDGPasses.h>

using namespace mlir;

class MemRefInsider : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider>
{
};

template<typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<PtrElementModel<T>, T>
{
};

int
main(int argc, char * argv[])
{
  mlir::registerAllPasses();
  registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::rvsdg::RVSDGDialect>();
  registry.insert<mlir::jlm::JLMDialect>();
  registry.insert<mlir::polygeist::PolygeistDialect>();

  registry.addExtension(+[](MLIRContext * ctx, LLVM::LLVMDialect * dialect)
                        {
                          LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
                        });
  registry.addExtension(+[](MLIRContext * ctx, LLVM::LLVMDialect * dialect)
                        {
                          LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
                        });
  registry.addExtension(+[](MLIRContext * ctx, LLVM::LLVMDialect * dialect)
                        {
                          LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
                        });
  registry.addExtension(+[](MLIRContext * ctx, LLVM::LLVMDialect * dialect)
                        {
                          LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
                        });
  registry.addExtension(+[](MLIRContext * ctx, memref::MemRefDialect * dialect)
                        {
                          MemRefType::attachInterface<PtrElementModel<MemRefType>>(*ctx);
                        });

  registry.addExtension(
      +[](MLIRContext * ctx, LLVM::LLVMDialect * dialect)
      {
        LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(*ctx);
      });

  registry.addExtension(
      +[](MLIRContext * ctx, LLVM::LLVMDialect * dialect)
      {
        LLVM::LLVMPointerType::attachInterface<PtrElementModel<LLVM::LLVMPointerType>>(*ctx);
      });

  registry.addExtension(
      +[](MLIRContext * ctx, LLVM::LLVMDialect * dialect)
      {
        LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(*ctx);
      });

  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "RVSDG Optimizer driver", registry));
}
