#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "RVSDG/RVSDGDialect.h"
#include "RVSDG/RVSDGPasses.h"
#include <mlir/Pass/PassManager.h>
#include <unordered_set>
#include <vector>

// Custom hash function for mlir::Value using the opaque pointer
namespace std
{
template<>
struct hash<mlir::Value>
{
  std::size_t
  operator()(const mlir::Value & v) const
  {
    return std::hash<void *>()(v.getAsOpaquePointer());
  }
};

template<>
struct equal_to<mlir::Value>
{
  bool
  operator()(const mlir::Value & lhs, const mlir::Value & rhs) const
  {
    return lhs.getAsOpaquePointer() == rhs.getAsOpaquePointer();
  }
};
}

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
    return true;
  }

  void
  runOnOperation() override
  {
    printf("Running ImportPolygeistPass\n");
    auto op = getOperation();
    auto module = mlir::dyn_cast<mlir::ModuleOp>(op);
    getUsedValues(module);

    // for (auto & [op, dependencies] : operationDependencies)
    // {
    //   printf("Operation with dependencies:\n");
    //   op->dump();
    //   printf("Dependencies:\n");
    //   for (auto dependency : dependencies)
    //   {
    //     dependency.dump();
    //   }
    //   printf("\n");
    // }
  }

  void
  getUsedValues(mlir::Operation * op)
  {
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op)) // Handle the module case
    {
      for (auto & operation : module->getRegion(0).getOps())
      {
        getUsedValues(&operation);
      }

      return;
    }

    auto dependencies = op->getOperands();
    for (auto dependency : dependencies)
    {
      auto definingRegion = dependency.getParentRegion();
      auto currentRegion = op->getParentRegion();
      while (definingRegion != currentRegion)
      {
        auto currentOp = currentRegion->getParentOp();
        if (operationDependencies.find(currentOp) == operationDependencies.end())
        {
          operationDependencies[currentOp] = std::unordered_set<mlir::Value>();
        }
        operationDependencies[currentOp].insert(dependency);
        currentRegion = currentOp->getParentRegion();
      }
    }

    for (auto & region : op->getRegions())
    {
      for (auto & op : region.getOps())
      {
        getUsedValues(&op);
      }
    }
  }

private:
  bool nestedPMInitialized = false;
  mlir::OpPassManager nestedPM;
  std::map<mlir::Operation *, std::unordered_set<mlir::Value>> operationDependencies;

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
