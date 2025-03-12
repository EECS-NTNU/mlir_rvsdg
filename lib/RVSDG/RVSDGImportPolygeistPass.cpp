#include "RVSDG/RVSDGPasses.h"
#include <mlir/Pass/PassManager.h>

namespace mlir::rvsdg::importPolygeistPass
{
#define GEN_PASS_DEF_RVSDG_IMPORTPOLYGEISTPASS
#include "RVSDG/Passes.h.inc"
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
    // return opName.isAnyOf("arith.addf");
  }

  void
  runOnOperation() override
  {
    mlir::Operation * op = getOperation();
    //     mlir::OpPassManager pm;
    //     pm.addPass(mlir::createConvertPolygeistToRVSDGPass());
    //     if (failed(runPipeline(pm, op)))
    //     {
    //         return signalPassFailure();
    // }

    for (auto & region : op->getRegions())
    {
      for (auto & nested_op : region.getOps())
      {
        // std::cout << nested_op.getName().getStringRef().data() << std::endl;
      }
    }
  }

private:
};

std::unique_ptr<::mlir::Pass>
createRVSDG_ImportPolygeistPass()
{
  auto pass = std::make_unique<ImportPolygeistPass>();
  return pass;
}
