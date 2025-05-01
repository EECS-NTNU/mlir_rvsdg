#include "JLM/JLMDialect.h"
#include "JLM/JLMOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Dialect.h"
#include "polygeist/Ops.h"
#include "RVSDG/RVSDGDialect.h"
#include "RVSDG/RVSDGPasses.h"
#include <iostream>
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
  {}

  ImportPolygeistPass(const ImportPolygeistPass & pass)
  {}

  void
  getDependentDialects(mlir::DialectRegistry & registry) const override
  {
    registry.insert<mlir::rvsdg::RVSDGDialect>();
    registry.insert<mlir::jlm::JLMDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::vector::VectorDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::polygeist::PolygeistDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

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

    printf("Getting used values\n");
    getUsedValues(module);
    printf("Sorting globals\n");
    sortGlobals(module);
    printf("Converting module\n");

    // Create a new builder for the module's context
    mlir::OpBuilder moduleBuilder(&context);

    auto omega = ConvertModule(module, moduleBuilder);
    printf("Converted module\n");

    // Clear the module's attributes and get its block
    module->setAttrs(mlir::ArrayRef<mlir::NamedAttribute>{});
    auto & block = module.getRegion().getBlocks().front();

    // Clear the block and add omega
    block.clear();
    block.push_back(omega);

    // Verify the module
    if (failed(mlir::verify(module)))
    {
      llvm::errs() << "Module verification failed after adding omega\n";
      signalPassFailure();
      return;
    }

    printf("Done\n");
  }

  mlir::Type
  ConvertType(mlir::Type type, mlir::OpBuilder & builder)
  {
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type))
    {
      return builder.getType<::mlir::LLVM::LLVMPointerType>();
    }
    else if (auto funcType = mlir::dyn_cast<mlir::FunctionType>(type))
    {
      auto convertedInputs = std::vector<mlir::Type>();
      for (auto input : funcType.getInputs())
      {
        convertedInputs.push_back(ConvertType(input, builder));
      }
      auto convertedOutputs = std::vector<mlir::Type>();
      for (auto output : funcType.getResults())
      {
        convertedOutputs.push_back(ConvertType(output, builder));
      }
      return builder.getType<mlir::FunctionType>(convertedInputs, convertedOutputs);
    }
    else if (auto funcType = mlir::dyn_cast<mlir::LLVM::LLVMFunctionType>(type))
    {
      auto convertedInputs = std::vector<mlir::Type>();
      for (auto input : funcType.getParams())
      {
        convertedInputs.push_back(ConvertType(input, builder));
      }
      auto convertedOutputs = std::vector<mlir::Type>();
      for (auto output : funcType.getReturnTypes())
      {
        convertedOutputs.push_back(ConvertType(output, builder));
      }
      return builder.getType<mlir::FunctionType>(convertedInputs, convertedOutputs);
    }
    return type;
  }

  ::llvm::SmallVector<::mlir::Value>
  GetConvertedInputs(
      mlir::Operation & op,
      const std::unordered_map<mlir::Value, mlir::Value> & valueMap)
  {
    ::llvm::SmallVector<::mlir::Value> inputs;
    for (const auto & operand : op.getOperands())
    {
      inputs.push_back(ConvertValue(operand, valueMap));
    }
    return inputs;
  }

  mlir::Value
  ConvertValue(mlir::Value value, const std::unordered_map<mlir::Value, mlir::Value> & valueMap)
  {
    auto it = valueMap.find(value);
    if (it != valueMap.end())
    {
      return it->second;
    }
    printf("Could not find value: \n");
    value.dump();
    assert(false);
  }

  mlir::Value
  ConvertName(std::string name, const std::unordered_map<std::string, mlir::Value> & nameMap)
  {
    auto it = nameMap.find(name);
    if (it != nameMap.end())
    {
      assert(it->second);
      return it->second;
    }
    printf("Could not find name: \n");
    printf("%s\n", name.c_str());
  }

  // Helper function for calculating memory offsets for memrefs
  mlir::Value
  CalculateMemrefOffset(
      mlir::Value memref,
      mlir::MemRefType memrefType,
      mlir::ValueRange indices,
      const std::unordered_map<mlir::Value, mlir::Value> & valueMap,
      mlir::Block & resultBlock,
      mlir::OpBuilder & builder)
  {
    auto nDims = memrefType.getShape().size();
    assert(indices.size() == nDims);

    // Handle the case where nDims = 0 (scalar memref) by returning a constant 0
    if (nDims == 0)
    {
      return ConvertValue(memref, valueMap);
    }

    mlir::Value currentOutput;
    size_t stride = 1;
    for (int i = nDims - 1; i >= 0; i--)
    {
      auto index = ConvertValue(indices[i], valueMap);
      auto intIndex = builder.create<mlir::arith::IndexCastOp>(
          builder.getUnknownLoc(),
          builder.getIntegerType(64),
          index);
      resultBlock.push_back(intIndex);
      auto strideOp = builder.create<mlir::arith::ConstantIntOp>(
          builder.getUnknownLoc(),
          stride,
          builder.getIntegerType(64));
      resultBlock.push_back(strideOp);
      auto multiplyOp = builder.create<mlir::arith::MulIOp>(
          builder.getUnknownLoc(),
          builder.getType<mlir::IntegerType>(64),
          intIndex,
          strideOp);
      resultBlock.push_back(multiplyOp);
      if (i == nDims - 1)
      {
        currentOutput = multiplyOp;
      }
      else
      {
        auto addOp = builder.create<mlir::arith::AddIOp>(
            builder.getUnknownLoc(),
            builder.getType<mlir::IntegerType>(64),
            currentOutput,
            multiplyOp);
        resultBlock.push_back(addOp);
        currentOutput = addOp;
      }
      stride *= memrefType.getShape()[i];
    }

    auto gepOp = builder.create<mlir::LLVM::GEPOp>(
        builder.getUnknownLoc(),
        builder.getType<mlir::LLVM::LLVMPointerType>(),
        memrefType.getElementType(),
        ConvertValue(memref, valueMap),
        currentOutput);
    resultBlock.push_back(gepOp);
    return gepOp;
  }

  ::llvm::SmallVector<::mlir::Value>
  ConvertRegion(
      mlir::Region & resultRegion,
      mlir::Region & sourceRegion,
      // bool includeMemState,
      // bool includeIOState,
      std::unordered_map<mlir::Value, mlir::Value> & oldValueMap,
      std::unordered_map<std::string, mlir::Value> & oldNameMap,
      mlir::OpBuilder & builder)
  {
    printf("Converting region\n");
    std::unordered_map<mlir::Value, mlir::Value> valueMap;
    std::unordered_map<std::string, mlir::Value> nameMap;
    auto & resultBlock = resultRegion.getBlocks().front();

    for (auto oldValue : oldValueMap)
    {
      valueMap[oldValue.first] = oldValue.second;
    }
    for (auto oldName : oldNameMap)
    {
      nameMap[oldName.first] = oldName.second;
    }

    // for (auto globalName : nameDependencies[sourceRegion.getParentOp()])
    // {
    //   auto global = globals[globalName];
    //   printf("Adding global:%s\n", globalName.c_str());
    //   // global.dump();
    //   resultBlock.addArgument(ConvertType(global.getType()), Builder_->getUnknownLoc());
    //   nameMap[globalName] = resultBlock.getArguments().back();
    // }

    // Create an MLIR operation for each RVSDG node and store each pair in a
    // hash map for easy lookup of corresponding MLIR operation
    for (auto & op : sourceRegion.getOps())
    {
      // op.dump();
      auto inputs = GetConvertedInputs(op, valueMap);

      auto convertedOp = ConvertOperation(op, resultBlock, valueMap, nameMap, inputs, builder);
      if (!convertedOp)
      {
        continue;
      }
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
      std::unordered_map<mlir::Value, mlir::Value> & valueMap,
      std::unordered_map<std::string, mlir::Value> & nameMap,
      llvm::SmallVector<mlir::Value> & inputs,
      mlir::OpBuilder & builder)
  {
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
    {
      printf("Converting scf.for to Theta\n");

      llvm::SmallVector<mlir::Value> thetaArgs = { inputs.begin(), inputs.end() };
      llvm::SmallVector<mlir::Value> thetaBlockArgsOld = { forOp.getOperands().begin(),
                                                           forOp.getOperands().end() };
      for (auto dependency : operationDependencies[&op])
      {
        thetaArgs.push_back(ConvertValue(dependency, valueMap));
        thetaBlockArgsOld.push_back(dependency);
      }
      thetaBlockArgsOld[0] =
          forOp.getInductionVar(); // These are used for block arguments and need the induction var
                                   // rather than the start value

      auto thetaNameDependenciesSet = nameDependencies[&op];
      auto thetaNameDependencies = llvm::SmallVector<std::string>(
          thetaNameDependenciesSet.begin(),
          thetaNameDependenciesSet.end());
      auto nameDependenciesStartIndex = thetaArgs.size();
      for (auto name : thetaNameDependencies)
      {
        thetaArgs.push_back(ConvertName(name, nameMap));
      }

      thetaArgs.push_back(newestMemState);
      thetaArgs.push_back(newestIOState);

      llvm::SmallVector<mlir::Type> thetaOutputs = {};
      for (auto arg : thetaArgs)
      {
        thetaOutputs.push_back(arg.getType());
      }

      llvm::SmallVector<mlir::NamedAttribute> attributes = {};
      auto theta = builder.create<mlir::rvsdg::ThetaNode>(
          builder.getUnknownLoc(),
          thetaOutputs,
          thetaArgs,
          attributes);

      auto & thetaBlock = theta.getRegion().emplaceBlock();
      for (auto arg : thetaArgs)
      {
        thetaBlock.addArgument(arg.getType(), builder.getUnknownLoc());
      }

      auto nThetaArgs = thetaArgs.size();
      auto thetaBlockMemState = thetaBlock.getArgument(nThetaArgs - 2);
      auto thetaBlockIOState = thetaBlock.getArgument(nThetaArgs - 1);
      newestMemState = thetaBlockMemState;
      newestIOState = thetaBlockIOState;

      auto thetaBlockInductionVar = thetaBlock.getArgument(0);
      auto thetaBlockUpperBound = thetaBlock.getArgument(1);
      auto thetaBlockStep = thetaBlock.getArgument(2);

      auto predicate = builder.create<mlir::arith::CmpIOp>(
          builder.getUnknownLoc(),
          mlir::arith::CmpIPredicate::slt,
          thetaBlockInductionVar,
          thetaBlockUpperBound);
      thetaBlock.push_back(predicate);

      ::llvm::SmallVector<::mlir::Attribute> mappingVector = {
        ::mlir::rvsdg::MatchRuleAttr::get(builder.getContext(), ::llvm::ArrayRef<int64_t>(0), 0),
        ::mlir::rvsdg::MatchRuleAttr::get(builder.getContext(), ::llvm::ArrayRef<int64_t>(1), 1)
      };

      auto match = builder.create<mlir::rvsdg::Match>(
          builder.getUnknownLoc(),
          builder.getType<mlir::rvsdg::RVSDG_CTRLType>(2),
          predicate,
          mlir::ArrayAttr::get(builder.getContext(), mappingVector));
      thetaBlock.push_back(match);

      // Create gamma node (if-then-else) for first iteration check
      auto gamma = builder.create<mlir::rvsdg::GammaNode>(
          builder.getUnknownLoc(),
          thetaOutputs,
          match,
          thetaBlock.getArguments(),
          2);

      auto & gammaBlock = gamma.getRegion(0).emplaceBlock();
      auto & dummyBlock = gamma.getRegion(1).emplaceBlock();

      for (auto arg : gamma.getInputs())
      {
        gammaBlock.addArgument(arg.getType(), builder.getUnknownLoc());
        dummyBlock.addArgument(arg.getType(), builder.getUnknownLoc());
      }
      auto blockArgsValueMap = std::unordered_map<mlir::Value, mlir::Value>();
      auto blockArgsNameMap = std::unordered_map<std::string, mlir::Value>();
      for (size_t i = 0; i < thetaBlockArgsOld.size(); i++)
      {
        blockArgsValueMap[thetaBlockArgsOld[i]] = gammaBlock.getArgument(i);
      }
      for (size_t i = nameDependenciesStartIndex; i < thetaArgs.size() - 2; i++)
      {
        blockArgsNameMap[thetaNameDependencies[i - nameDependenciesStartIndex]] =
            gammaBlock.getArgument(i);
      }

      newestMemState = gammaBlock.getArgument(gammaBlock.getArguments().size() - 2);
      newestIOState = gammaBlock.getArgument(gammaBlock.getArguments().size() - 1);
      ConvertRegion(
          gamma.getRegion(0),
          forOp.getRegion(),
          blockArgsValueMap,
          blockArgsNameMap,
          builder);

      auto gammaResult = builder.create<mlir::rvsdg::GammaResult>(
          builder.getUnknownLoc(),
          gammaBlock.getArguments());

      auto dummyResult = builder.create<mlir::rvsdg::GammaResult>(
          builder.getUnknownLoc(),
          dummyBlock.getArguments());

      gammaBlock.push_back(gammaResult);
      dummyBlock.push_back(dummyResult);

      thetaBlock.push_back(gamma);

      auto thetaResult = builder.create<mlir::rvsdg::ThetaResult>(
          builder.getUnknownLoc(),
          match,
          thetaBlock.getArguments());

      thetaBlock.push_back(thetaResult);

      newestMemState = theta.getOutputs()[theta.getOutputs().size() - 2];
      newestIOState = theta.getOutputs()[theta.getOutputs().size() - 1];

      resultBlock.push_back(theta);
      return theta;
    }
    else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
    {
      printf("Converting scf.if to Gamma\n");

      llvm::SmallVector<mlir::Value> gammaArgs = { inputs.begin(), inputs.end() }; // Op args
      llvm::SmallVector<mlir::Value> gammaArgsOld = { ifOp.getOperand() };         // Old op args
      for (auto dependency : operationDependencies[&op]) // Op Dependencies
      {
        gammaArgs.push_back(ConvertValue(dependency, valueMap));
        gammaArgsOld.push_back(dependency);
      }

      auto gammaNameDependenciesSet = nameDependencies[&op];
      auto gammaNameDependencies = llvm::SmallVector<std::string>(
          gammaNameDependenciesSet.begin(),
          gammaNameDependenciesSet.end());
      auto nameDependenciesStartIndex = gammaArgs.size();
      for (auto name : gammaNameDependencies)
      {
        gammaArgs.push_back(ConvertName(name, nameMap));
      }
      gammaArgs.push_back(newestMemState);
      gammaArgs.push_back(newestIOState);

      llvm::SmallVector<mlir::Type> gammaOutputs = {};
      for (auto arg : gammaArgs)
      {
        gammaOutputs.push_back(arg.getType());
      }

      ::llvm::SmallVector<::mlir::Attribute> mappingVector = {
        ::mlir::rvsdg::MatchRuleAttr::get(builder.getContext(), ::llvm::ArrayRef<int64_t>(0), 0),
        ::mlir::rvsdg::MatchRuleAttr::get(builder.getContext(), ::llvm::ArrayRef<int64_t>(1), 1)
      };

      auto predicate = inputs[0];
      auto match = builder.create<mlir::rvsdg::Match>(
          builder.getUnknownLoc(),
          builder.getType<mlir::rvsdg::RVSDG_CTRLType>(2),
          predicate,
          mlir::ArrayAttr::get(builder.getContext(), mappingVector));
      resultBlock.push_back(match);

      llvm::SmallVector<mlir::NamedAttribute> attributes = {};
      auto gamma = builder.create<mlir::rvsdg::GammaNode>(
          builder.getUnknownLoc(),
          gammaOutputs,
          match,
          gammaArgs,
          2);

      auto & ifBlock = gamma.getRegion(0).emplaceBlock();
      auto & elseBlock = gamma.getRegion(1).emplaceBlock();

      for (auto arg : gamma.getInputs()) // Does not include predicate
      {
        ifBlock.addArgument(arg.getType(), builder.getUnknownLoc());
        elseBlock.addArgument(arg.getType(), builder.getUnknownLoc());
      }

      auto ifBlockArgsValueMap = std::unordered_map<mlir::Value, mlir::Value>();
      auto ifBlockArgsNameMap = std::unordered_map<std::string, mlir::Value>();
      auto elseBlockArgsValueMap = std::unordered_map<mlir::Value, mlir::Value>();
      auto elseBlockArgsNameMap = std::unordered_map<std::string, mlir::Value>();
      for (size_t i = 0; i < gammaArgsOld.size(); i++)
      {
        ifBlockArgsValueMap[gammaArgsOld[i]] = ifBlock.getArgument(i);
        elseBlockArgsValueMap[gammaArgsOld[i]] = elseBlock.getArgument(i);
      }
      for (size_t i = nameDependenciesStartIndex; i < gammaArgs.size() - 2; i++)
      {
        ifBlockArgsNameMap[gammaNameDependencies[i - nameDependenciesStartIndex]] =
            ifBlock.getArgument(i);
        elseBlockArgsNameMap[gammaNameDependencies[i - nameDependenciesStartIndex]] =
            elseBlock.getArgument(i);
      }

      auto gammaBlockMemStateIndex = ifBlock.getArguments().size() - 2;
      auto gammaBlockIOStateIndex = ifBlock.getArguments().size() - 1;

      newestMemState = ifBlock.getArguments()[gammaBlockMemStateIndex];
      newestIOState = ifBlock.getArguments()[gammaBlockIOStateIndex];
      ConvertRegion(
          gamma.getRegion(0),
          ifOp.getRegion(0),
          ifBlockArgsValueMap,
          ifBlockArgsNameMap,
          builder);

      newestMemState = elseBlock.getArguments()[gammaBlockMemStateIndex];
      newestIOState = elseBlock.getArguments()[gammaBlockIOStateIndex];
      ConvertRegion(
          gamma.getRegion(1),
          ifOp.getRegion(1),
          elseBlockArgsValueMap,
          elseBlockArgsNameMap,
          builder);

      auto ifResult =
          builder.create<mlir::rvsdg::GammaResult>(builder.getUnknownLoc(), ifBlock.getArguments());

      auto elseResult = builder.create<mlir::rvsdg::GammaResult>(
          builder.getUnknownLoc(),
          elseBlock.getArguments());

      ifBlock.push_back(ifResult);
      elseBlock.push_back(elseResult);

      newestMemState = gamma.getOutputs()[gamma.getOutputs().size() - 2];
      newestIOState = gamma.getOutputs()[gamma.getOutputs().size() - 1];

      resultBlock.push_back(gamma);
      return gamma;
    }
    else if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(op))
    {
      assert(yieldOp.getOperands().size() == 0 && "Yield op with operands is not supported");
      return nullptr; // TODO: Thetaresult should be created here, but none of the loops actually
                      // yield anything
    }
    else if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op))
    {
      printf("Converting func.func to Lambda or OmegaArgument\n");

      if (funcOp.empty())
      {
        auto type = ConvertType(funcOp.getFunctionType(), builder);
        auto omegaArgument = builder.create<::mlir::rvsdg::OmegaArgument>(
            builder.getUnknownLoc(),
            type,
            type,
            builder.getStringAttr("external_linkage"),
            builder.getStringAttr(funcOp.getName()));
        nameMap[funcOp.getName().str()] = omegaArgument;
        // globals[funcOp.getName().str()] = omegaArgument;
        resultBlock.push_back(omegaArgument);
        return omegaArgument;
      }
      else
      {
        auto contextVarsSet = operationDependencies[&op];
        auto contextNamesSet = nameDependencies[&op];

        auto contextVars =
            llvm::SmallVector<mlir::Value>(contextVarsSet.begin(), contextVarsSet.end());
        auto contextNames =
            llvm::SmallVector<std::string>(contextNamesSet.begin(), contextNamesSet.end());

        llvm::SmallVector<mlir::Value> lambdaArgs = {}; // Op args
        for (auto dependency : contextVars)             // Op Dependencies
        {
          lambdaArgs.push_back(ConvertValue(dependency, valueMap));
        }
        auto lambdaNamesStartIndex = lambdaArgs.size();
        for (auto name : contextNames)
        {
          printf("Adding lambda arg name:%s\n", name.c_str());
          lambdaArgs.push_back(ConvertName(name, nameMap));
        }

        ::llvm::SmallVector<::mlir::NamedAttribute> attributes;
        auto symbolName = builder.getNamedAttr(
            builder.getStringAttr("sym_name"),
            builder.getStringAttr(funcOp.getName()));
        attributes.push_back(symbolName);
        auto linkage = builder.getNamedAttr(
            builder.getStringAttr("linkage"),
            builder.getStringAttr("external_linkage"));
        attributes.push_back(linkage);

        auto lambda = builder.create<::mlir::rvsdg::LambdaNode>(
            builder.getUnknownLoc(),
            ConvertType(funcOp.getFunctionType(), builder),
            lambdaArgs,
            ::llvm::ArrayRef<::mlir::NamedAttribute>(attributes));

        auto & lambdaBlock = lambda.getRegion().emplaceBlock();
        auto blockArgsValueMap = std::unordered_map<mlir::Value, mlir::Value>();
        auto blockArgsNameMap = std::unordered_map<std::string, mlir::Value>();
        printf("Lambda block function args: %d\n", funcOp.getArguments().size());
        for (auto arg : funcOp.getArguments())
        {
          lambdaBlock.addArgument(ConvertType(arg.getType(), builder), builder.getUnknownLoc());
          blockArgsValueMap[arg] = lambdaBlock.getArguments().back();
        }

        lambdaBlock.addArgument(
            builder.getType<mlir::rvsdg::MemStateEdgeType>(),
            builder.getUnknownLoc());
        newestMemState =
            lambdaBlock.getArguments()
                .back(); // These should not need to be reset, as the func ops are located in the
                         // module region, where there are no memstates or iostates
        lambdaBlock.addArgument(
            builder.getType<mlir::rvsdg::IOStateEdgeType>(),
            builder.getUnknownLoc());
        newestIOState = lambdaBlock.getArguments().back();

        printf("Lambda block context args: %d\n", contextVars.size());
        for (auto dependency : contextVars)
        {
          lambdaBlock.addArgument(
              ConvertType(dependency.getType(), builder),
              builder.getUnknownLoc());
          blockArgsValueMap[dependency] = lambdaBlock.getArguments().back();
        }
        printf("Lambda block context names: %d\n", contextNames.size());
        for (auto name : contextNames)
        {
          lambdaBlock.addArgument(
              ConvertType(nameMap[name].getType(), builder),
              builder.getUnknownLoc());
          blockArgsNameMap[name] = lambdaBlock.getArguments().back();
        }

        ConvertRegion(
            lambda.getRegion(),
            funcOp.getRegion(),
            blockArgsValueMap,
            blockArgsNameMap,
            builder);
        // auto lambdaResult =
        //     Builder_->create<::mlir::rvsdg::LambdaResult>(Builder_->getUnknownLoc(),
        //     regionResults);
        // lambdaBlock.push_back(lambdaResult);
        // globals[funcOp.getName().str()] = lambda;
        nameMap[funcOp.getName().str()] = lambda;
        resultBlock.push_back(lambda);
        return lambda;
      }
    }
    else if (auto globalOp = mlir::dyn_cast<mlir::LLVM::GlobalOp>(op))
    {
      if (globalOp.getConstantAttr())
      {
        auto value = globalOp.getValue();
        assert(value.has_value());

        auto delta = builder.create<::mlir::rvsdg::DeltaNode>(
            builder.getUnknownLoc(),
            builder.getType<::mlir::LLVM::LLVMPointerType>(),
            inputs,
            llvm::StringRef(globalOp.getName()),
            llvm::StringRef("private_linkage"),
            llvm::StringRef(""),
            true);
        // resultBlock.push_back(delta);
        auto & deltaBlock = delta.getRegion().emplaceBlock();
        // globals[globalOp.getName().str()] = delta;
        // printf("Adding global:%s\n", globalOp.getName().str().c_str());
        nameMap[globalOp.getName().str()] = delta;
        auto stringAttr = value.value().dyn_cast<mlir::StringAttr>();
        llvm::SmallVector<mlir::Value> charArray;
        if (stringAttr)
        {
          for (auto c : stringAttr.getValue())
          {
            auto constantChar = builder.create<mlir::arith::ConstantIntOp>(
                builder.getUnknownLoc(),
                c,
                builder.getIntegerType(8));
            deltaBlock.push_back(constantChar);
            charArray.push_back(constantChar);
          }
        }

        auto constantDataArray = builder.create<mlir::jlm::ConstantDataArray>(
            builder.getUnknownLoc(),
            builder.getType<mlir::LLVM::LLVMArrayType>(builder.getI8Type(), charArray.size()),
            charArray);
        deltaBlock.push_back(constantDataArray);

        auto deltaResult =
            builder.create<::mlir::rvsdg::DeltaResult>(builder.getUnknownLoc(), constantDataArray);
        deltaBlock.push_back(deltaResult);
        resultBlock.push_back(delta);
        return delta;
      }
      else
      {
        auto type = ConvertType(globalOp.getType(), builder);
        auto importedType = type.isa<mlir::LLVM::LLVMFunctionType>()
                              ? type
                              : builder.getType<::mlir::LLVM::LLVMPointerType>();
        auto omegaArgument = builder.create<::mlir::rvsdg::OmegaArgument>(
            builder.getUnknownLoc(),
            importedType,
            type,
            builder.getStringAttr("external_linkage"),
            builder.getStringAttr(globalOp.getName()));
        printf("Converting llvm.mlir.global external to OmegaArgument\n");
        // globals[globalOp.getName().str()] = omegaArgument;
        nameMap[globalOp.getName().str()] = omegaArgument;
        resultBlock.push_back(omegaArgument);
        return omegaArgument;
      }
    }
    else if (auto funcOp = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op))
    {
      printf("Converting llvm.func to OmegaArgument\n");
      auto type = ConvertType(funcOp.getFunctionType(), builder);
      auto omegaArgument = builder.create<::mlir::rvsdg::OmegaArgument>(
          builder.getUnknownLoc(),
          type,
          type,
          builder.getStringAttr("external_linkage"),
          builder.getStringAttr(funcOp.getName()));
      // globals[funcOp.getName().str()] = omegaArgument;
      nameMap[funcOp.getName().str()] = omegaArgument;
      resultBlock.push_back(omegaArgument);
      return omegaArgument;
    }
    else if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op))
    {
      auto type = allocOp.getType().cast<mlir::MemRefType>();
      auto elementSize = type.getElementTypeBitWidth();
      auto nElements = type.getNumElements();
      auto sizeOp = builder.create<mlir::arith::ConstantIntOp>(
          builder.getUnknownLoc(),
          elementSize * nElements,
          builder.getIntegerType(64));
      resultBlock.push_back(sizeOp);
      auto malloc = builder.create<mlir::jlm::Malloc>(
          builder.getUnknownLoc(),
          builder.getType<::mlir::LLVM::LLVMPointerType>(),
          builder.getType<::mlir::rvsdg::MemStateEdgeType>(),
          sizeOp.getResult());
      mlir::Value mallocMemState = malloc.getOutputMemState();
      auto memorystatemerge = builder.create<mlir::rvsdg::MemStateMerge>(
          builder.getUnknownLoc(),
          builder.getType<::mlir::rvsdg::MemStateEdgeType>(),
          mlir::ValueRange{ mallocMemState, newestMemState });
      newestMemState = memorystatemerge.getOutput();
      resultBlock.push_back(malloc); // Memory state merge depends on malloc, but the outputs of
                                     // malloc are used for output mapping
      resultBlock.push_back(memorystatemerge);
      return malloc;
      printf("Converting memref.alloc to jlm.malloc\n");
      // Change to call to malloc (jlm.malloc)
    }
    else if (auto allocaOp = mlir::dyn_cast<mlir::memref::AllocaOp>(op))
    {
      auto type = allocaOp.getType().cast<mlir::MemRefType>();
      auto valueType = ConvertType(type.getElementType(), builder);
      auto elementSize = type.getElementTypeBitWidth();
      auto nElements = type.getNumElements();
      auto sizeOp = builder.create<mlir::arith::ConstantIntOp>(
          builder.getUnknownLoc(),
          elementSize * nElements,
          builder.getIntegerType(64));
      resultBlock.push_back(sizeOp);
      auto alloca = builder.create<mlir::jlm::Alloca>(
          builder.getUnknownLoc(),
          builder.getType<::mlir::LLVM::LLVMPointerType>(),
          builder.getType<::mlir::rvsdg::MemStateEdgeType>(),
          valueType,
          sizeOp.getResult(),
          (uint32_t)allocaOp.getAlignment().value_or(1),
          mlir::ValueRange{ newestMemState });
      newestMemState = alloca.getOutputMemState();
      resultBlock.push_back(alloca);
      return alloca;
      printf("Converting memref.alloca to jlm.malloc\n");
    }
    else if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    {
      printf("Converting memref.store to jlm.store\n");

      // Calculate offset using helper function
      auto ptr = CalculateMemrefOffset(
          storeOp.getMemRef(),
          storeOp.getMemRefType(),
          storeOp.getIndices(),
          valueMap,
          resultBlock,
          builder);

      auto store = builder.create<mlir::jlm::Store>(
          builder.getUnknownLoc(),
          builder.getType<::mlir::rvsdg::MemStateEdgeType>(),
          ptr,
          ConvertValue(storeOp.getValueToStore(), valueMap),
          1,
          mlir::ValueRange{ newestMemState });

      newestMemState = store.getOutputMemState();
      resultBlock.push_back(store);
      return store;
      // Change to jlm.store
    }
    else if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    {
      printf("Converting memref.load to jlm.load\n");
      auto ptr = CalculateMemrefOffset(
          loadOp.getMemRef(),
          loadOp.getMemRefType(),
          loadOp.getIndices(),
          valueMap,
          resultBlock,
          builder);

      auto load = builder.create<mlir::jlm::Load>(
          builder.getUnknownLoc(),
          ConvertType(loadOp.getResult().getType(), builder),
          builder.getType<::mlir::rvsdg::MemStateEdgeType>(),
          ptr,
          1,
          mlir::ValueRange{ newestMemState });

      newestMemState = load.getOutputMemState();
      resultBlock.push_back(load);
      return load;
      // Change to jlm.load
    }
    else if (auto castOp = mlir::dyn_cast<mlir::memref::CastOp>(op))
    {
      auto input = ConvertValue(castOp.getOperand(), valueMap);
      valueMap[castOp.getResult()] = input;
      printf("Removing memref.cast\n");
      return nullptr;
      // Remove
    }
    else if (auto pointer2memrefOp = mlir::dyn_cast<mlir::polygeist::Pointer2MemrefOp>(op))
    {
      auto input = ConvertValue(pointer2memrefOp.getOperand(), valueMap);
      valueMap[pointer2memrefOp.getResult()] = input;
      printf("Removing polygeist.pointer2memref\n");
      return nullptr;
      // Remove
    }
    else if (auto memref2pointerOp = mlir::dyn_cast<mlir::polygeist::Memref2PointerOp>(op))
    {
      auto input = ConvertValue(memref2pointerOp.getOperand(), valueMap);
      valueMap[memref2pointerOp.getResult()] = input;
      printf("Removing polygeist.memref2pointer\n");
      return nullptr;
      // Remove
    }
    else if (auto addressOfOp = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(op))
    {
      // auto global = globals[addressOfOp.getGlobalName().str()];
      auto global = ConvertName(addressOfOp.getGlobalName().str(), nameMap);
      valueMap[addressOfOp.getResult()] = global;
      printf("Removing llvm.mlir.addressof\n");
      return nullptr;
      // Might be able to remove
    }
    else if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op))
    {
      mlir::SmallVector<mlir::Type> resultTypes;
      for (auto result : callOp.getResults())
      {
        resultTypes.push_back(ConvertType(result.getType(), builder));
      }
      resultTypes.push_back(builder.getType<mlir::rvsdg::IOStateEdgeType>());
      resultTypes.push_back(builder.getType<mlir::rvsdg::MemStateEdgeType>());
      printf("Converting func.call to jlm.call\n");
      auto call = builder.create<mlir::jlm::Call>(
          builder.getUnknownLoc(),
          resultTypes,
          ConvertName(callOp.getCallee().str(), nameMap),
          inputs,
          newestIOState,
          newestMemState);
      newestIOState = call.getOutputIoState();
      newestMemState = call.getOutputMemState();
      resultBlock.push_back(call);
      return call;
      // Change to jlm.call
    }
    else if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op))
    {
      printf("Converting llvm.call to jlm.call\n");
      mlir::SmallVector<mlir::Type> resultTypes;
      for (auto result : callOp.getResults())
      {
        resultTypes.push_back(ConvertType(result.getType(), builder));
      }
      resultTypes.push_back(builder.getType<mlir::rvsdg::IOStateEdgeType>());
      resultTypes.push_back(builder.getType<mlir::rvsdg::MemStateEdgeType>());
      auto callee = callOp.getCallee();
      assert(callee.has_value());
      auto call = builder.create<mlir::jlm::Call>(
          builder.getUnknownLoc(),
          resultTypes,
          ConvertName(callee.value().str(), nameMap),
          inputs,
          newestIOState,
          newestMemState);
      newestIOState = call.getOutputIoState();
      newestMemState = call.getOutputMemState();
      resultBlock.push_back(call);
      return call;
      // Change to jlm.call
    }
    else if (auto deallocOp = mlir::dyn_cast<mlir::memref::DeallocOp>(op))
    {
      printf("Converting memref.dealloc to jlm.free\n");
      auto free = builder.create<mlir::jlm::Free>(
          builder.getUnknownLoc(),
          llvm::SmallVector<mlir::Type>{ builder.getType<mlir::rvsdg::MemStateEdgeType>() },
          builder.getType<mlir::rvsdg::IOStateEdgeType>(),
          ConvertValue(deallocOp.getOperand(), valueMap),
          mlir::ValueRange{ newestMemState },
          newestIOState);
      newestMemState = free.getOutputMemStates()[0];
      newestIOState = free.getOutputIoState();
      resultBlock.push_back(free);
      return free;
      // Change to jlm.free
    }
    else if (auto loadOp = mlir::dyn_cast<mlir::LLVM::LoadOp>(op))
    {
      printf("Converting llvm.load to jlm.load\n");
      auto load = builder.create<mlir::jlm::Load>(
          builder.getUnknownLoc(),
          ConvertType(loadOp.getResult().getType(), builder),
          builder.getType<::mlir::rvsdg::MemStateEdgeType>(),
          ConvertValue(loadOp.getOperand(), valueMap),
          0,
          mlir::ValueRange{ newestMemState });

      newestMemState = load.getOutputMemState();
      resultBlock.push_back(load);
      return load;
    }
    else if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op))
    {
      llvm::SmallVector<mlir::Value> returnValues = {};
      for (auto operand : returnOp.getOperands())
      {
        returnValues.push_back(ConvertValue(operand, valueMap));
      }
      auto lambdaResult =
          builder.create<mlir::rvsdg::LambdaResult>(builder.getUnknownLoc(), returnValues);
      resultBlock.push_back(lambdaResult);
      return lambdaResult;
    }
    else if (
        mlir::isa<mlir::arith::ConstantOp>(op) || mlir::isa<mlir::arith::IndexCastOp>(op)
        || mlir::isa<mlir::arith::SIToFPOp>(op) || mlir::isa<mlir::arith::MulIOp>(op)
        || mlir::isa<mlir::arith::DivFOp>(op) || mlir::isa<mlir::arith::AddFOp>(op)
        || mlir::isa<mlir::arith::SubFOp>(op) || mlir::isa<mlir::math::SqrtOp>(op)
        || mlir::isa<mlir::arith::CmpFOp>(op) || mlir::isa<mlir::arith::CmpIOp>(op)
        || mlir::isa<mlir::arith::RemSIOp>(op) || mlir::isa<mlir::arith::RemUIOp>(op)
        || mlir::isa<mlir::arith::AddIOp>(op) || mlir::isa<mlir::arith::SelectOp>(op)
        || mlir::isa<mlir::arith::MulFOp>(op) || mlir::isa<mlir::LLVM::GEPOp>(op)
        || mlir::isa<mlir::arith::SubIOp>(op) || mlir::isa<mlir::arith::DivUIOp>(op)
        || mlir::isa<mlir::arith::DivSIOp>(op) || mlir::isa<mlir::arith::AndIOp>(op))
    {
      printf("Keeping operation: %s\n", op.getName().getStringRef().data());
      auto clonedOp = op.clone();
      clonedOp->setOperands(inputs);
      resultBlock.push_back(clonedOp);
      return clonedOp;
    }
    else
    {
      std::cerr << "Unhandled operation: " << op.getName().getStringRef().data() << "\n";
      assert(false && "unhandled operation");
    }
  }

  mlir::rvsdg::OmegaNode
  ConvertModule(mlir::ModuleOp module, mlir::OpBuilder & builder)
  {
    // Initialize memory and IO state
    newestMemState = nullptr;
    newestIOState = nullptr;

    // Create the omega operation using the provided builder
    auto omega = builder.create<::mlir::rvsdg::OmegaNode>(builder.getUnknownLoc());
    auto & omegaBlock = omega.getRegion().emplaceBlock();

    std::unordered_map<mlir::Value, mlir::Value> valueMap;
    std::unordered_map<std::string, mlir::Value> nameMap;

    // Use the provided builder for conversion
    ConvertRegion(omega.getRegion(), module.getRegion(), valueMap, nameMap, builder);

    auto omegaResult = builder.create<::mlir::rvsdg::OmegaResult>(
        builder.getUnknownLoc(),
        mlir::ValueRange{}); // TODO: OmegaResult
    omegaBlock.push_back(omegaResult);

    return omega;
  }

  void
  sortGlobals(mlir::ModuleOp module)
  {
    // Step 1: Collect all symbol-producing ops (func.func and global ops)
    std::vector<mlir::Operation *> symbolOps;
    std::unordered_map<std::string, mlir::Operation *> symbolNameToOp;

    for (auto & op : module.getRegion().getOps())
    {
      if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(&op))
      {
        symbolOps.push_back(&op);
        symbolNameToOp[funcOp.getSymName().str()] = &op;
      }
      else if (auto globalOp = llvm::dyn_cast<mlir::LLVM::GlobalOp>(&op))
      {
        symbolOps.push_back(&op);
        symbolNameToOp[globalOp.getSymName().str()] = &op;
      }
      else if (auto funcOp = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(&op))
      {
        symbolOps.push_back(&op);
        symbolNameToOp[funcOp.getSymName().str()] = &op;
      }
      // Add more symbol-producing ops here if needed
    }

    // Step 2: Build the dependency graph
    // For each symbol op, find which other symbols it uses
    std::unordered_map<mlir::Operation *, std::vector<mlir::Operation *>> dependencies;
    for (auto * op : symbolOps)
    {
      std::vector<mlir::Operation *> deps;

      // Check for func.func calls
      if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op))
      {
        funcOp.walk(
            [&](mlir::Operation * innerOp)
            {
              if (auto callOp = llvm::dyn_cast<mlir::func::CallOp>(innerOp))
              {
                auto callee = callOp.getCallee().str();
                if (symbolNameToOp.count(callee))
                {
                  deps.push_back(symbolNameToOp[callee]);
                }
                else
                {
                  llvm::errs() << "Callee " << callee << " not found in symbolNameToOp\n";
                }
              }
              else if (auto addressOf = llvm::dyn_cast<mlir::LLVM::AddressOfOp>(innerOp))
              {
                auto globalName = addressOf.getGlobalName().str();
                if (symbolNameToOp.count(globalName))
                {
                  deps.push_back(symbolNameToOp[globalName]);
                }
                else
                {
                  llvm::errs() << "Global " << globalName << " not found in symbolNameToOp\n";
                }
              }
            });
      }
      dependencies[op] = std::move(deps);
    }

    // Step 3: Topological sort
    std::vector<mlir::Operation *> sorted;
    std::unordered_set<mlir::Operation *> visited, visiting;

    std::function<void(mlir::Operation *)> dfs = [&](mlir::Operation * op)
    {
      if (visited.count(op))
        return;
      if (visiting.count(op))
      {
        llvm::errs() << "Cyclic dependency detected in globals/functions!\n";
        return;
      }
      visiting.insert(op);
      for (auto * dep : dependencies[op])
      {
        dfs(dep);
      }
      visiting.erase(op);
      visited.insert(op);
      sorted.push_back(op);
    };

    for (auto * op : symbolOps)
    {
      dfs(op);
    }

    // Step 4: Reorder the ops in the module region
    // Remove all symbol ops from the region, then re-insert in sorted order
    auto & block = module.getRegion().front();
    for (auto * op : symbolOps)
    {
      op->remove();
    }
    for (auto * op : llvm::reverse(sorted))
    {
      block.getOperations().push_front(op);
    }
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

    if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op))
    {
      auto calleeName = callOp.getCallee().str();
      auto currentRegion = op->getParentRegion();
      while (!mlir::isa<mlir::ModuleOp>(currentRegion->getParentOp()))
      {
        auto currentOp = currentRegion->getParentOp();
        if (nameDependencies.find(currentOp) == nameDependencies.end())
        {
          nameDependencies[currentOp] = std::unordered_set<std::string>();
        }
        nameDependencies[currentOp].insert(calleeName);
        currentRegion = currentOp->getParentRegion();
      }
    }

    if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op))
    {
      auto calleeName = callOp.getCallee();
      assert(calleeName.has_value());
      auto currentRegion = op->getParentRegion();
      while (!mlir::isa<mlir::ModuleOp>(currentRegion->getParentOp()))
      {
        auto currentOp = currentRegion->getParentOp();
        if (nameDependencies.find(currentOp) == nameDependencies.end())
        {
          nameDependencies[currentOp] = std::unordered_set<std::string>();
        }
        nameDependencies[currentOp].insert(calleeName.value().str());
        currentRegion = currentOp->getParentRegion();
      }
    }

    if (auto addressOfOp = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(op))
    {
      auto addrName = addressOfOp.getGlobalName();
      auto currentRegion = op->getParentRegion();
      while (!mlir::isa<mlir::ModuleOp>(currentRegion->getParentOp()))
      {
        auto currentOp = currentRegion->getParentOp();
        if (nameDependencies.find(currentOp) == nameDependencies.end())
        {
          nameDependencies[currentOp] = std::unordered_set<std::string>();
        }
        nameDependencies[currentOp].insert(addrName.str());
        currentRegion = currentOp->getParentRegion();
      }
    }

    llvm::SmallVector<mlir::Value> dependencies = { op->getOperands().begin(),
                                                    op->getOperands().end() };
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
  std::map<mlir::Operation *, std::unordered_set<std::string>> nameDependencies;
  std::map<mlir::Value, mlir::Value> currentDependencies;
  std::map<std::string, mlir::Value> globals;
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
