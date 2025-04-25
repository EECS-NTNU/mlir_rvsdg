#include "JLM/JLMDialect.h"
#include "JLM/JLMOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
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

  ImportPolygeistPass()
      : Context_(std::make_unique<mlir::MLIRContext>())
  {
    Context_->getOrLoadDialect<mlir::rvsdg::RVSDGDialect>();
    Context_->getOrLoadDialect<mlir::jlm::JLMDialect>();
    Context_->getOrLoadDialect<mlir::arith::ArithDialect>();
    Context_->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    Builder_ = std::make_unique<mlir::OpBuilder>(Context_.get());
  }

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
    auto & context = *op->getContext();

    mlir::PassManager preTransformPm(&context);
    preTransformPm.addPass(mlir::createLowerAffinePass());
    preTransformPm.run(module);

    module.dump();
    printf("Getting used values\n");
    getUsedValues(module);
    printf("Converting module\n");
    ConvertModule(module);

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

  mlir::Type
  ConvertType(mlir::Type type)
  {
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type))
    {
      return Builder_->getType<::mlir::LLVM::LLVMPointerType>();
    }
    else if (auto funcType = mlir::dyn_cast<mlir::FunctionType>(type))
    {
      auto convertedInputs = std::vector<mlir::Type>();
      for (auto input : funcType.getInputs())
      {
        convertedInputs.push_back(ConvertType(input));
      }
      auto convertedOutputs = std::vector<mlir::Type>();
      for (auto output : funcType.getResults())
      {
        convertedOutputs.push_back(ConvertType(output));
      }
      return Builder_->getType<mlir::FunctionType>(convertedInputs, convertedOutputs);
    }
    return type;
  }

  ::llvm::SmallVector<::mlir::Value>
  GetConvertedInputs(
      mlir::Operation & op,
      const std::unordered_map<mlir::Value, mlir::Value> & valueMap)
  {
    ::llvm::SmallVector<::mlir::Value> inputs;
    for (auto operand : op.getOperands())
    {
      auto it = valueMap.find(operand);
      if (it != valueMap.end())
      {
        inputs.push_back(it->second);
      }
      else
      {
        printf("Unimplemented input type: \n");
        operand.dump();
      }
    }
    return inputs;
  }

  ::llvm::SmallVector<::mlir::Value>
  ConvertRegion(
      mlir::Region & resultRegion,
      mlir::Region & sourceRegion,
      bool includeMemState,
      bool includeIOState)
  {
    printf("Converting region\n");
    std::unordered_map<mlir::Value, mlir::Value> valueMap;
    auto & resultBlock = resultRegion.getBlocks().front();

    if (includeMemState)
    {
      resultBlock.addArgument(
          Builder_->getType<::mlir::rvsdg::MemStateEdgeType>(),
          Builder_->getUnknownLoc());
      newestMemState = resultBlock.getArguments().back();
    }
    if (includeIOState)
    {
      resultBlock.addArgument(
          Builder_->getType<::mlir::rvsdg::IOStateEdgeType>(),
          Builder_->getUnknownLoc());
      newestIOState = resultBlock.getArguments().back();
    }

    for (auto arg : sourceRegion.getArguments())
    {
      resultBlock.addArgument(ConvertType(arg.getType()), Builder_->getUnknownLoc());
      valueMap[arg] = resultBlock.getArguments().back();
    }

    for (auto additionalArg : operationDependencies[sourceRegion.getParentOp()])
    {
      resultBlock.addArgument(ConvertType(additionalArg.getType()), Builder_->getUnknownLoc());
      valueMap[additionalArg] = resultBlock.getArguments().back();
    }

    // Create an MLIR operation for each RVSDG node and store each pair in a
    // hash map for easy lookup of corresponding MLIR operation
    for (auto & op : sourceRegion.getOps())
    {
      auto inputs = GetConvertedInputs(op, valueMap);

      auto convertedOp = ConvertOperation(op, resultBlock, inputs);
      for (size_t i = 0; i < op.getNumResults(); i++)
      {
        valueMap[op.getResult(i)] = convertedOp->getResult(i);
      }
    }

    return {};
  }

  mlir::Operation *
  ConvertOperation(
      mlir::Operation & op,
      mlir::Block & resultBlock,
      llvm::SmallVector<mlir::Value> & inputs)
  {
    op.dump();
    ::mlir::Operation * MlirOp;
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
    {
      printf("Converting scf.for to Theta\n");
      // Change to Theta
    }
    else if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op))
    {
      printf("Converting func.func to Lambda\n");

      ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
      auto symbolName = Builder_->getNamedAttr(
          Builder_->getStringAttr("sym_name"),
          Builder_->getStringAttr(funcOp.getName()));
      attributes.push_back(symbolName);
      auto linkage = Builder_->getNamedAttr(
          Builder_->getStringAttr("linkage"),
          Builder_->getStringAttr("external_linkage"));
      attributes.push_back(linkage);

      auto lambda = Builder_->create<::mlir::rvsdg::LambdaNode>(
          Builder_->getUnknownLoc(),
          ConvertType(funcOp.getFunctionType()),
          inputs,
          ::llvm::ArrayRef<::mlir::NamedAttribute>(attributes));
      // resultBlock.push_back(lambda);

      auto & lambdaBlock = lambda.getRegion().emplaceBlock();
      auto regionResults = ConvertRegion(lambda.getRegion(), funcOp.getRegion(), true, true);
      auto lambdaResult =
          Builder_->create<::mlir::rvsdg::LambdaResult>(Builder_->getUnknownLoc(), regionResults);
      lambdaBlock.push_back(lambdaResult);
      MlirOp = lambda;
    }
    else if (auto globalOp = mlir::dyn_cast<mlir::LLVM::GlobalOp>(op))
    {
      if (globalOp.getConstantAttr())
      {
        auto value = globalOp.getValue();
        assert(value.has_value());

        auto delta = Builder_->create<::mlir::rvsdg::DeltaNode>(
            Builder_->getUnknownLoc(),
            Builder_->getType<::mlir::LLVM::LLVMPointerType>(),
            inputs,
            llvm::StringRef(globalOp.getName()),
            llvm::StringRef("private_linkage"),
            llvm::StringRef(""),
            true);
        // resultBlock.push_back(delta);
        auto & deltaBlock = delta.getRegion().emplaceBlock();

        auto stringAttr = value.value().dyn_cast<mlir::StringAttr>();
        llvm::SmallVector<mlir::Value> charArray;
        if (stringAttr)
        {
          for (auto c : stringAttr.getValue())
          {
            auto constantChar = Builder_->create<mlir::arith::ConstantIntOp>(
                Builder_->getUnknownLoc(),
                c,
                Builder_->getIntegerType(8));
            deltaBlock.push_back(constantChar);
            charArray.push_back(constantChar);
          }
        }

        auto constantDataArray = Builder_->create<mlir::jlm::ConstantDataArray>(
            Builder_->getUnknownLoc(),
            Builder_->getType<mlir::LLVM::LLVMArrayType>(Builder_->getI8Type(), charArray.size()),
            charArray);
        deltaBlock.push_back(constantDataArray);

        auto deltaResult = Builder_->create<::mlir::rvsdg::DeltaResult>(
            Builder_->getUnknownLoc(),
            constantDataArray);
        deltaBlock.push_back(deltaResult);
        MlirOp = delta;
      }
      else
      {
        auto type = ConvertType(globalOp.getType());
        auto importedType = type.isa<mlir::LLVM::LLVMFunctionType>()
                              ? type
                              : Builder_->getType<::mlir::LLVM::LLVMPointerType>();
        auto omegaArgument = Builder_->create<::mlir::rvsdg::OmegaArgument>(
            Builder_->getUnknownLoc(),
            importedType,
            type,
            Builder_->getStringAttr("external_linkage"),
            Builder_->getStringAttr(globalOp.getName()));
        printf("Converting llvm.mlir.global external to OmegaArgument\n");
        MlirOp = omegaArgument;
      }
    }
    else if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op))
    {
      auto type = allocOp.getType().cast<mlir::MemRefType>();
      auto elementSize = type.getElementTypeBitWidth();
      auto nElements = type.getNumElements();
      auto sizeOp = Builder_->create<mlir::arith::ConstantIntOp>(
          Builder_->getUnknownLoc(),
          elementSize * nElements,
          Builder_->getIntegerType(64));
      auto malloc = Builder_->create<mlir::jlm::Malloc>(
          Builder_->getUnknownLoc(),
          Builder_->getType<::mlir::LLVM::LLVMPointerType>(),
          Builder_->getType<::mlir::rvsdg::MemStateEdgeType>(),
          sizeOp.getResult());
      mlir::Value mallocMemState = malloc.getOutputMemState();
      auto memorystatemerge = Builder_->create<mlir::rvsdg::MemStateMerge>(
          Builder_->getUnknownLoc(),
          Builder_->getType<::mlir::rvsdg::MemStateEdgeType>(),
          mlir::ValueRange{ mallocMemState, newestMemState });
      newestMemState = memorystatemerge.getOutput();
      MlirOp = malloc;
      printf("Converting memref.alloc to jlm.malloc\n");
      // Change to call to malloc (jlm.malloc)
    }
    else if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    {
      printf("Converting memref.store to jlm.store\n");
      // Change to jlm.store
    }
    else if (auto castOp = mlir::dyn_cast<mlir::memref::CastOp>(op))
    {
      printf("Removing memref.cast\n");
      // Remove
    }
    else if (auto cmpOp = mlir::dyn_cast<mlir::arith::CmpFOp>(op))
    {
      printf("Keeping arith.cmpf\n");
      // Might keep
    }
    else if (op.getName().getStringRef() == "polygeist.pointer2memref")
    {
      printf("Removing polygeist.pointer2memref\n");
      // Remove
    }
    else if (auto cmpOp = mlir::dyn_cast<mlir::arith::CmpIOp>(op))
    {
      printf("Keeping arith.cmpi\n");
      // Might keep
    }
    else if (auto addressOfOp = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(op))
    {
      printf("Considering removal of llvm.mlir.addressof\n");
      // Might be able to remove
    }
    else if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op))
    {
      printf("Converting func.call to jlm.call\n");
      // Change to jlm.call
    }
    else if (auto deallocOp = mlir::dyn_cast<mlir::memref::DeallocOp>(op))
    {
      printf("Converting memref.dealloc to jlm.free\n");
      // Change to jlm.free
    }
    else if (auto loadOp = mlir::dyn_cast<mlir::LLVM::LoadOp>(op))
    {
      printf("Converting llvm.load to jlm.load\n");
      // Change to jlm.load
    }
    else if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    {
      printf("Converting memref.load to jlm.load\n");
      // Change to jlm.load
    }
    else if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op))
    {
      printf("Converting return to LambdaResult\n");
      // Change to LambdaResult
    }
    else if (
        mlir::isa<mlir::arith::ConstantOp>(op) || mlir::isa<mlir::arith::IndexCastOp>(op)
        || mlir::isa<mlir::arith::SIToFPOp>(op) || mlir::isa<mlir::arith::MulIOp>(op)
        || mlir::isa<mlir::arith::DivFOp>(op) || mlir::isa<mlir::arith::AddFOp>(op)
        || mlir::isa<mlir::arith::SubFOp>(op))
    {
      printf("Keeping arithmetic operation: %s\n", op.getName().getStringRef().data());
      MlirOp = op.clone();
    }
    else if (auto sqrtOp = mlir::dyn_cast<mlir::math::SqrtOp>(op))
    {
      printf("Keeping math.sqrt\n");
      MlirOp = op.clone();
      // Keep
    }
    else if (auto gepOp = mlir::dyn_cast<mlir::LLVM::GEPOp>(op))
    {
      printf("Keeping llvm.getelementptr\n");
      MlirOp = op.clone();
      // Keep
    }
    else if (auto remOp = mlir::dyn_cast<mlir::arith::RemSIOp>(op))
    {
      printf("Keeping arith.remsi\n");
      MlirOp = op.clone();
      // Keep
    }
    else if (auto funcOp = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op))
    {
      printf("Converting llvm.func to OmegaArgument\n");
      auto type = ConvertType(funcOp.getFunctionType());
      auto omegaArgument = Builder_->create<::mlir::rvsdg::OmegaArgument>(
          Builder_->getUnknownLoc(),
          type,
          type,
          Builder_->getStringAttr("external_linkage"),
          Builder_->getStringAttr(funcOp.getName()));
      MlirOp = omegaArgument;
    }
    else if (auto selectOp = mlir::dyn_cast<mlir::arith::SelectOp>(op))
    {
      printf("Maybe converting arith.select to Gamma\n");
      // Maybe change to Gamma
    }
    else
    {
      printf("Unhandled operation: %s\n", op.getName().getStringRef().data());
    }
    if (!MlirOp)
    {
      printf("Unhandled operation: %s\n", op.getName().getStringRef().data());
      assert(false && "Unhandled operation");
    }
    resultBlock.push_back(MlirOp);
    return MlirOp;
    // assert(false && "Unhandled operation");
  }

  mlir::rvsdg::OmegaNode
  ConvertModule(mlir::ModuleOp module)
  {
    auto omega = Builder_->create<::mlir::rvsdg::OmegaNode>(Builder_->getUnknownLoc());
    auto & omegaBlock = omega.getRegion().emplaceBlock();

    auto region = ConvertRegion(omega.getRegion(), module.getRegion(), false, false);
    // auto omegaResult =
    //     Builder_->create<::mlir::rvsdg::OmegaResult>(Builder_->getUnknownLoc(), regionResults);
    // omegaBlock.push_back(omegaResult);

    return omega;
  }

  void
  getUsedValues(mlir::Operation * op)
  {
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op)) // Handle the module case
    {
      for (auto & operation : module.getRegion().getOps())
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
  std::unique_ptr<::mlir::OpBuilder> Builder_;
  std::unique_ptr<::mlir::MLIRContext> Context_;
  mlir::Value newestMemState;
  mlir::Value newestIOState;

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
