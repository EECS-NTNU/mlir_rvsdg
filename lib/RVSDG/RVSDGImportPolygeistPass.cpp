#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "RVSDG/RVSDGDialect.h"
#include "RVSDG/RVSDGPasses.h"
#include <mlir/Pass/PassManager.h>

namespace mlir::rvsdg::importPolygeistPass
{
#define GEN_PASS_DEF_RVSDG_IMPORTPOLYGEISTPASS
#include "RVSDG/Passes.h.inc"
#include "RVSDG/Patterns.h.inc"

// Helper function to convert LLVM.mlir.global to RVSDG deltanode
static Value
convertLLVMGlobalToRVSDGDelta(
    PatternRewriter & rewriter,
    Operation * op,
    Attribute symNameAttr,
    Attribute valueAttr)
{
  auto loc = op->getLoc();
  auto globalOp = cast<LLVM::GlobalOp>(op);

  auto deltaNode = rewriter.create<mlir::rvsdg::DeltaNode>(loc);
  Region & deltaRegion = deltaNode.getBody();
  auto * block = rewriter.createBlock(&deltaRegion);

  // Position the insertion point at the end of the block
  rewriter.setInsertionPointToEnd(block);

  // Get the global value
  Value globalValue;
  if (valueAttr && !valueAttr.isa<UnitAttr>())
  {
    // If the global has an initializer, use it
    globalValue =
        rewriter.create<LLVM::ConstantOp>(loc, globalOp.getType(), valueAttr);
  }
  else
  {
    // Otherwise, create a zero value
    globalValue = rewriter.create<LLVM::UndefOp>(loc, globalOp.getType());
  }

  // Create the delta result operation
  rewriter.create<mlir::rvsdg::DeltaResult>(loc, globalValue);

  // Return the deltanode
  return deltaNode.getOutput();
}

}

struct ImportPolygeistPass
    : mlir::rvsdg::importPolygeistPass::impl::RVSDG_ImportPolygeistPassBase<ImportPolygeistPass>
{

  ImportPolygeistPass() = default;

  ImportPolygeistPass(const ImportPolygeistPass & pass)
  {}

  bool
  canScheduleOn(mlir::RegisteredOperationName opName) const override
  {
    return true;
  }

  void
  runOnOperation() override
  {
    mlir::RewritePatternSet patterns(&getContext());

    // Add the patterns generated from the TableGen definitions
    mlir::rvsdg::importPolygeistPass::populateWithGenerated(patterns);

    // Apply the patterns
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    {
      signalPassFailure();
    }
  }

private:
  bool nestedPMInitialized = false;
  mlir::OpPassManager nestedPM;

  inline mlir::OpPassManager
  initNestedPassManager()
  {
    mlir::OpPassManager nestedPM;
    nestedPM.addPass(createRVSDG_ImportPolygeistPass());
    return nestedPM;
  }
};

std::unique_ptr<::mlir::Pass>
createRVSDG_ImportPolygeistPass()
{
  auto pass = std::make_unique<ImportPolygeistPass>();
  return pass;
}
