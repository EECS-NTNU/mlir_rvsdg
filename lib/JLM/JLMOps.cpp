#include <unordered_set>

#include "mlir/IR/Block.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "RVSDG/RVSDGASMDirectives.h"

#include "JLM/JLMOps.h"

using namespace mlir;
using namespace jlm;

/*
 * Alignment is a hardware guarantee, so an impossible value is a bug rather than an unknown.
 * LLVM's own global rejects anything that is not a positive power of two, and that is the rule
 * every operation carrying an alignment follows.
 */
static LogicalResult verifyAlignment(Operation * op, int64_t alignment) {
  if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
    return op->emitOpError("has an alignment of ")
        << alignment << ", which is not a positive power of two.";
  }
  return LogicalResult::success();
}

/**
 * Memory operators
 */

/**
 * Load
 */
LogicalResult jlm::Load::verify() {
  // isVolatile must agree with the presence of the threaded I/O state operands.
  auto inputIoState = this->getInputIoState();
  auto outputIoState = this->getOutputIoState();

  if (getIsVolatile())
  {
    if (!inputIoState || !outputIoState)
      return emitOpError(
          "volatile load requires an input and an output I/O state (inputIoState/outputIoState)");
  }
  else if (inputIoState || outputIoState)
  {
    return emitOpError(
        "non-volatile load must not have an input or output I/O state (inputIoState/outputIoState)");
  }

  if (failed(verifyAlignment(*this, this->getAlignment())))
    return failure();

  return LogicalResult::success();
}

/*
* Store
*/
LogicalResult jlm::Store::verify() {
  // isVolatile must agree with the presence of the threaded I/O state operands.
  auto inputIoState = this->getInputIoState();
  auto outputIoState = this->getOutputIoState();

  if (getIsVolatile())
  {
    if (!inputIoState || !outputIoState)
      return emitOpError(
          "volatile store requires an input and an output I/O state (inputIoState/outputIoState)");
  }
  else if (inputIoState || outputIoState)
  {
    return emitOpError(
        "non-volatile store must not have an input or output I/O state (inputIoState/outputIoState)");
  }

  if (failed(verifyAlignment(*this, this->getAlignment())))
    return failure();

  return LogicalResult::success();
}

/**
 * Alloca
 */
LogicalResult jlm::Alloca::verify() {
  return verifyAlignment(*this, this->getAlignment());
}

/**
 * ConstantDataArray
 */
LogicalResult jlm::ConstantDataArray::verify() {

  auto operands = this->getOperands().getTypes();
  if (!operands.empty()) {
    auto firstType = operands.front();
    for (auto operand : operands) {
      if (operand != firstType) {
        return emitOpError("All operands must have the same type.");
      }
    }
  }

  return LogicalResult::success();
}

/**
 * IOBarrier
 */
LogicalResult jlm::IOBarrier::verify() {
  auto inputType = this->getOperand(0).getType();
  auto outputType = this->getResult().getType();
  if (inputType != outputType) {
    return emitOpError("Input and output types must be the same.");
  }
  return LogicalResult::success();
}

/**
 * MemoryHoistBarrier
 */
LogicalResult jlm::MemoryHoistBarrier::verify() {
  if (!llvm::isa<LLVM::LLVMPointerType>(this->getAddress().getType()))
    return emitOpError("address operand must have LLVM pointer type.");
  if (!llvm::isa<LLVM::LLVMPointerType>(this->getOutput().getType()))
    return emitOpError("result must have LLVM pointer type.");
  return LogicalResult::success();
}

/**
 * Memcpy
 */
LogicalResult jlm::Memcpy::verify() {
  // isVolatile must agree with the presence of the threaded I/O state operands.
  auto inputIoState = this->getInputIoState();
  auto outputIoState = this->getOutputIoState();

  if (getIsVolatile())
  {
    if (!inputIoState || !outputIoState)
      return emitOpError(
          "volatile memcpy requires an input and an output I/O state (inputIoState/outputIoState)");
  }
  else if (inputIoState || outputIoState)
  {
    return emitOpError(
        "non-volatile memcpy must not have an input or output I/O state (inputIoState/outputIoState)");
  }

  return LogicalResult::success();
}

/**
 * Auto generated sources
 */
#define GET_OP_CLASSES
#include "JLM/Ops.cpp.inc"

/**
 * Implement dialect method for registering Ops
 */
void mlir::jlm::JLMDialect::addJLMOps() {
  addOperations<
#define GET_OP_LIST
#include "JLM/Ops.cpp.inc"
      >();
}
