#include <iostream>
#include <unordered_set>
#include <vector>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Pass/PassManager.h>

#include <JLM/JLMDialect.h>
#include <JLM/JLMOps.h>
#include <polygeist/Dialect.h>
#include <polygeist/Ops.h>
#include <RVSDG/RVSDGDialect.h>
#include <RVSDG/RVSDGPasses.h>

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
    auto op = getOperation();
    auto module = mlir::dyn_cast<mlir::ModuleOp>(op);
    auto & context = *op->getContext();

    mlir::PassManager preTransformPm(&context);
    preTransformPm.addPass(mlir::createLowerAffinePass());
    auto logicalResult = preTransformPm.run(module);

    SortGlobals(module);
    GetUsedValues(module);

    // Create a new builder for the module's context
    mlir::OpBuilder moduleBuilder(&context);

    auto omega = ConvertModule(module, moduleBuilder);

    module->setAttrs(mlir::ArrayRef<mlir::NamedAttribute>{});

    auto & block = module.getRegion().getBlocks().front();
    block.clear();
    block.push_back(omega);

    if (failed(mlir::verify(module)))
    {
      std::cerr << "Module verification failed after adding omega\n";
      module.dump();
      signalPassFailure();
      return;
    }
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
      return ConvertFunctionType(funcType.getInputs(), funcType.getResults(), builder);
    }
    else if (auto funcType = mlir::dyn_cast<mlir::LLVM::LLVMFunctionType>(type))
    {
      return ConvertFunctionType(funcType.getParams(), funcType.getReturnTypes(), builder, funcType.isVarArg());
    }
    return type;
  }

  // Helper method to convert function types
  mlir::Type
  ConvertFunctionType(mlir::TypeRange inputs, mlir::TypeRange outputs, mlir::OpBuilder & builder, bool isVariadic=false, bool addStateEdges=false)
  {
    auto convertedInputs = std::vector<mlir::Type>();
    for (auto input : inputs)
    {
      convertedInputs.push_back(ConvertType(input, builder));
    }
    if (isVariadic)
    {
      convertedInputs.push_back(builder.getType<::mlir::jlm::VarargListType>());
    }
    if ( addStateEdges )
    {
      convertedInputs.push_back(builder.getType<mlir::rvsdg::IOStateEdgeType>());
      convertedInputs.push_back(builder.getType<mlir::rvsdg::MemStateEdgeType>());
    }
    auto convertedOutputs = std::vector<mlir::Type>();
    for (auto output : outputs)
    {
      convertedOutputs.push_back(ConvertType(output, builder));
    }
    if ( addStateEdges )
    {
      convertedOutputs.push_back(builder.getType<mlir::rvsdg::IOStateEdgeType>());
      convertedOutputs.push_back(builder.getType<mlir::rvsdg::MemStateEdgeType>());
    }
    return builder.getType<mlir::FunctionType>(convertedInputs, convertedOutputs);
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
    std::cerr << "Could not find value: \n";
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
    std::cerr << "Could not find name: " << name << "\n";
    assert(false);
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

    // When there are no dimensions, the memref points to a single value
    // so we just return pointer to that value
    if (nDims == 0)
    {
      return ConvertValue(memref, valueMap);
    }

    mlir::Value currentOutput;
    size_t stride = 1;
    for (ssize_t i = nDims - 1; i >= 0; i--)
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

  void
  ConvertRegion(
      mlir::Block & resultBlock,
      mlir::Region & sourceRegion,
      std::unordered_map<mlir::Value, mlir::Value> valueMap,
      std::unordered_map<std::string, mlir::Value> nameMap,
      mlir::OpBuilder & builder)
  {

    // Convert each operation in the source region
    for (auto & op : sourceRegion.getOps())
    {
      auto inputs = GetConvertedInputs(op, valueMap);

      auto convertedOp = ConvertOperation(op, resultBlock, valueMap, nameMap, inputs, builder);
      // Some operations are simply deleted, they mutate the valueMap directly
      if (!convertedOp)
      {
        continue;
      }
      // Update the valueMap with the new outputs of the operation
      for (size_t i = 0; i < op.getNumResults(); i++)
      {
        valueMap[op.getResult(i)] = convertedOp->getResult(i);
      }
    }
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
      llvm::SmallVector<mlir::Value> thetaArgs = {
        std::next(inputs.begin(), 3),
        inputs.end()
      }; // The actual outputs of the loop need to be placed first
      thetaArgs.push_back(inputs[0]); // Lower bound
      thetaArgs.push_back(inputs[1]); // Upper bound
      thetaArgs.push_back(inputs[2]); // Step
      llvm::SmallVector<mlir::Value> thetaBlockArgsOld = { forOp.getRegionIterArgs().begin(),
                                                           forOp.getRegionIterArgs().end() };
      thetaBlockArgsOld.push_back(
          forOp.getInductionVar()); // Lower bound is mapped to induction var
      thetaBlockArgsOld.push_back(forOp.getUpperBound());
      thetaBlockArgsOld.push_back(forOp.getStep());

      for (auto dependency : operationDependencies[&op])
      {
        thetaArgs.push_back(ConvertValue(dependency, valueMap));
        thetaBlockArgsOld.push_back(dependency);
      }

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

      auto thetaBlockInductionVar = thetaBlock.getArgument(forOp.getNumRegionIterArgs());
      auto thetaBlockUpperBound = thetaBlock.getArgument(forOp.getNumRegionIterArgs() + 1);
      // TODO: Step must be added to the induction var

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

      // Need to add the argument representing the predicate
      gammaBlock.addArgument(builder.getType<::mlir::NoneType>(), builder.getUnknownLoc());
      dummyBlock.addArgument(builder.getType<::mlir::NoneType>(), builder.getUnknownLoc());
      for (auto arg : gamma.getInputs())
      {
        gammaBlock.addArgument(arg.getType(), builder.getUnknownLoc());
        dummyBlock.addArgument(arg.getType(), builder.getUnknownLoc());
      }
      auto blockArgsValueMap = std::unordered_map<mlir::Value, mlir::Value>();
      auto blockArgsNameMap = std::unordered_map<std::string, mlir::Value>();
      for (size_t i = 0; i < thetaBlockArgsOld.size(); i++)
      {
        // The first gamma argument is the predicate and not connected
        blockArgsValueMap[thetaBlockArgsOld[i]] = gammaBlock.getArgument(i+1);
      }
      for (size_t i = nameDependenciesStartIndex; i < thetaArgs.size() - 2; i++)
      {
        // The first gammaargument is the predicate and not connected
        blockArgsNameMap[thetaNameDependencies[i - nameDependenciesStartIndex]] =
            gammaBlock.getArgument(i+1);
      }

      newestMemState = gammaBlock.getArgument(gammaBlock.getArguments().size() - 2);
      newestIOState = gammaBlock.getArgument(gammaBlock.getArguments().size() - 1);
      ConvertRegion(gammaBlock, forOp.getRegion(), blockArgsValueMap, blockArgsNameMap, builder);

      // The first argument corresponds to the match and not part of the result
      auto dummyBlockArguments = dummyBlock.getArguments().drop_front();
      auto dummyResult = builder.create<mlir::rvsdg::GammaResult>(
          builder.getUnknownLoc(),
          dummyBlockArguments);

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
      llvm::SmallVector<mlir::Value> gammaArgs = { inputs.begin(), inputs.end() };
      llvm::SmallVector<mlir::Value> gammaArgsOld = { ifOp.getOperand() };
      for (auto dependency : operationDependencies[&op])
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
      for (auto result : ifOp.getResults())
      {
        gammaOutputs.push_back(ConvertType(result.getType(), builder));
      }
      gammaOutputs.push_back(newestMemState.getType());
      gammaOutputs.push_back(newestIOState.getType());

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

      // Need to add the argument representing the predicate
      ifBlock.addArgument(builder.getType<::mlir::NoneType>(), builder.getUnknownLoc());
      elseBlock.addArgument(builder.getType<::mlir::NoneType>(), builder.getUnknownLoc());
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
        // The predicate argument is not used so need to adjust indexing
        ifBlockArgsValueMap[gammaArgsOld[i]] = ifBlock.getArgument(i+1);
        elseBlockArgsValueMap[gammaArgsOld[i]] = elseBlock.getArgument(i+1);
      }
      for (size_t i = nameDependenciesStartIndex; i < gammaArgs.size() - 2; i++)
      {
        // The predicate argument is not used so need to adjust indexing
        ifBlockArgsNameMap[gammaNameDependencies[i - nameDependenciesStartIndex]] =
            ifBlock.getArgument(i+1);
        elseBlockArgsNameMap[gammaNameDependencies[i - nameDependenciesStartIndex]] =
            elseBlock.getArgument(i+1);
      }

      auto gammaBlockMemStateIndex = ifBlock.getArguments().size() - 2;
      auto gammaBlockIOStateIndex = ifBlock.getArguments().size() - 1;

      newestMemState = ifBlock.getArguments()[gammaBlockMemStateIndex];
      newestIOState = ifBlock.getArguments()[gammaBlockIOStateIndex];
      ConvertRegion(ifBlock, ifOp.getRegion(0), ifBlockArgsValueMap, ifBlockArgsNameMap, builder);

      newestMemState = elseBlock.getArguments()[gammaBlockMemStateIndex];
      newestIOState = elseBlock.getArguments()[gammaBlockIOStateIndex];
      if (!ifOp.getElseRegion().hasOneBlock())
      {
        elseBlock.push_back(builder.create<mlir::rvsdg::GammaResult>(
            builder.getUnknownLoc(),
            mlir::ValueRange({ newestMemState, newestIOState })));
      }
      else
      {
        ConvertRegion(
            elseBlock,
            ifOp.getRegion(1),
            elseBlockArgsValueMap,
            elseBlockArgsNameMap,
            builder);
      }

      newestMemState = gamma.getOutputs()[gamma.getOutputs().size() - 2];
      newestIOState = gamma.getOutputs()[gamma.getOutputs().size() - 1];

      resultBlock.push_back(gamma);
      return gamma;
    }

    else if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(op))
    {
      if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(yieldOp->getParentOp()))
      {
        auto results = llvm::SmallVector<mlir::Value>();
        for (auto result : yieldOp.getOperands())
        {
          results.push_back(ConvertValue(result, valueMap));
        }
        results.push_back(newestMemState);
        results.push_back(newestIOState);
        auto gammaResult =
            builder.create<mlir::rvsdg::GammaResult>(builder.getUnknownLoc(), results);
        resultBlock.push_back(gammaResult);
        return gammaResult;
      }
      else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(yieldOp->getParentOp()))
      {
        auto gamma = mlir::dyn_cast<mlir::rvsdg::GammaNode>(resultBlock.getParentOp());
        auto results = llvm::SmallVector<mlir::Value>();
        auto nYieldedValues = yieldOp.getOperands().size();
        for (auto result : yieldOp.getOperands())
        {
          results.push_back(ConvertValue(result, valueMap));
        }
        // The gamma needs to output the same types as it received as input
        // we just pass the values that are not yielded to the result
        for (size_t i = nYieldedValues; i < gamma.getNumResults(); i++)
        {
          // The first argument corresponds to the predicate and is not used in the result
          results.push_back(resultBlock.getArgument(i+1));
        }
        // Gamma result as we are inside the nested gamma
        auto gammaResult =
            builder.create<mlir::rvsdg::GammaResult>(builder.getUnknownLoc(), results);
        resultBlock.push_back(gammaResult);
        return gammaResult;
      }
      assert(false && "Yield in unsupported operation");
      return nullptr;
    }
    else if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op))
    {
      if (funcOp.empty())
      {
        // The function is not defined, so we create an OmegaArgument
        auto funcType = funcOp.getFunctionType();
        auto type = ConvertFunctionType(funcType.getInputs(), funcType.getResults(), builder, false, true);
        auto omegaArgument = builder.create<::mlir::rvsdg::OmegaArgument>(
            builder.getUnknownLoc(),
            type,
            type,
            builder.getStringAttr("external_linkage"),
            builder.getStringAttr(funcOp.getName()));
        nameMap[funcOp.getName().str()] = omegaArgument;
        resultBlock.push_back(omegaArgument);
        return omegaArgument;
      }
      else
      {
        // The function is defined, so we create a Lambda for it
        auto contextVarsSet = operationDependencies[&op];
        auto contextNamesSet = nameDependencies[&op];

        auto contextVars =
            llvm::SmallVector<mlir::Value>(contextVarsSet.begin(), contextVarsSet.end());
        auto contextNames =
            llvm::SmallVector<std::string>(contextNamesSet.begin(), contextNamesSet.end());

        llvm::SmallVector<mlir::Value> lambdaArgs = {};
        for (auto dependency : contextVars)
        {
          lambdaArgs.push_back(ConvertValue(dependency, valueMap));
        }
        for (auto name : contextNames)
        {
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
        for (auto dependency : contextVars)
        {
          lambdaBlock.addArgument(
              ConvertType(dependency.getType(), builder),
              builder.getUnknownLoc());
          blockArgsValueMap[dependency] = lambdaBlock.getArguments().back();
        }
        for (auto name : contextNames)
        {
          lambdaBlock.addArgument(
              ConvertType(nameMap[name].getType(), builder),
              builder.getUnknownLoc());
          blockArgsNameMap[name] = lambdaBlock.getArguments().back();
        }
        ConvertRegion(
            lambdaBlock,
            funcOp.getRegion(),
            blockArgsValueMap,
            blockArgsNameMap,
            builder);
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
        auto & deltaBlock = delta.getRegion().emplaceBlock();
        nameMap[globalOp.getName().str()] = delta;
        auto stringAttr = value.value().dyn_cast<mlir::StringAttr>();
        if (!stringAttr)
        {
          assert(false && "Constant global with type other than string is not supported");
        }
        llvm::SmallVector<mlir::Value> charArray;
        for (auto c : stringAttr.getValue())
        {
          auto constantChar = builder.create<mlir::arith::ConstantIntOp>(
              builder.getUnknownLoc(),
              c,
              builder.getIntegerType(8));
          deltaBlock.push_back(constantChar);
          charArray.push_back(constantChar);
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
        mlir::Type type;
        mlir::Type importedType;
        if ( auto funcType = mlir::dyn_cast<mlir::LLVM::LLVMFunctionType>(type) )
        {
          type = ConvertFunctionType(funcType.getParams(), funcType.getReturnTypes(), builder, funcType.isVarArg(), true);
          importedType = type;
        }
        else
        {
          type = ConvertType(globalOp.getType(), builder);
          importedType = builder.getType<::mlir::LLVM::LLVMPointerType>();
        }
        auto omegaArgument = builder.create<::mlir::rvsdg::OmegaArgument>(
            builder.getUnknownLoc(),
            importedType,
            type,
            builder.getStringAttr("external_linkage"),
            builder.getStringAttr(globalOp.getName()));
        nameMap[globalOp.getName().str()] = omegaArgument;
        resultBlock.push_back(omegaArgument);
        return omegaArgument;
      }
    }
    else if (auto funcOp = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op))
    {
      auto funcType = funcOp.getFunctionType();
      auto type = ConvertFunctionType(funcType.getParams(), funcType.getReturnTypes(), builder, funcType.isVarArg(), true);
      auto omegaArgument = builder.create<::mlir::rvsdg::OmegaArgument>(
          builder.getUnknownLoc(),
          type,
          type,
          builder.getStringAttr("external_linkage"),
          builder.getStringAttr(funcOp.getName()));
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
      resultBlock.push_back(malloc);
      resultBlock.push_back(memorystatemerge);
      return malloc;
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
    }
    else if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    {
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

      // Polygeist RVSDG only use a single memory state
      assert(std::size(store.getOutputMemStates()) == 1);
      newestMemState = store.getOutputMemStates().front();
      resultBlock.push_back(store);
      return store;
    }
    else if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    {
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

      // Polygeist RVSDG only use a single memory state
      assert(std::size(load.getOutputMemStates()) == 1);
      newestMemState = load.getOutputMemStates().front();
      resultBlock.push_back(load);
      return load;
    }
    else if (mlir::isa<mlir::memref::CastOp>(op) || mlir::isa<mlir::polygeist::Pointer2MemrefOp>(op) || mlir::isa<mlir::polygeist::Memref2PointerOp>(op))
    {
      valueMap[op.getResult(0)] = ConvertValue(op.getOperand(0), valueMap);
      return nullptr;
    }
    else if (auto addressOfOp = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(op))
    {
      valueMap[addressOfOp.getResult()] = ConvertName(addressOfOp.getGlobalName().str(), nameMap);
      return nullptr;
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
    }
    else if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op))
    {
      mlir::scf::ValueVector operands;
      auto functionType = callOp.getCalleeFunctionType();
      if ( functionType.isVarArg() )
      {
        auto signatureArguments = functionType.getNumParams();
        assert(signatureArguments <= inputs.size());
        for ( size_t i=0; i<signatureArguments; i++ )
        {
          operands.push_back(inputs[i]);
        }
        llvm::SmallVector<::mlir::Value> variadicInputs;
        for ( size_t i=signatureArguments; i < inputs.size(); i++ )
        {
          variadicInputs.push_back(inputs[i]);
        }
        auto varArgList = builder.create<::mlir::jlm::CreateVarArgList>(
          builder.getUnknownLoc(),
          builder.getType<::mlir::jlm::VarargListType>(),
          variadicInputs);
        resultBlock.push_back(varArgList);
        operands.push_back(varArgList);
      }
      else
      {
        operands = inputs;
      }

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
          operands,
          newestIOState,
          newestMemState);
      newestIOState = call.getOutputIoState();
      newestMemState = call.getOutputMemState();
      resultBlock.push_back(call);
      return call;
    }
    else if (auto deallocOp = mlir::dyn_cast<mlir::memref::DeallocOp>(op))
    {
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
    }
    else if (auto loadOp = mlir::dyn_cast<mlir::LLVM::LoadOp>(op))
    {
      auto load = builder.create<mlir::jlm::Load>(
          builder.getUnknownLoc(),
          ConvertType(loadOp.getResult().getType(), builder),
          builder.getType<::mlir::rvsdg::MemStateEdgeType>(),
          ConvertValue(loadOp.getOperand(), valueMap),
          0,
          mlir::ValueRange{ newestMemState });

      // Polygeist RVSDG only use a single memory state
      assert(std::size(load.getOutputMemStates()) == 1);
      newestMemState = load.getOutputMemStates().front();
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
      returnValues.push_back(newestMemState);
      returnValues.push_back(newestIOState);
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
    ConvertRegion(omegaBlock, module.getRegion(), valueMap, nameMap, builder);

    auto omegaResult = builder.create<::mlir::rvsdg::OmegaResult>(
        builder.getUnknownLoc(),
        mlir::ValueRange{}); // TODO: OmegaResult
    omegaBlock.push_back(omegaResult);

    return omega;
  }

  void
  SortGlobals(mlir::ModuleOp module)
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
  GetUsedValues(mlir::Operation * op)
  {
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op)) // Handle the module case
    {
      for (auto & operation : module.getRegion().getOps())
      {
        GetUsedValues(&operation);
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
        GetUsedValues(&op);
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
