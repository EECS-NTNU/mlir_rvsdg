#include <unordered_set>

#include "mlir/IR/Block.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "RVSDG/RVSDGASMDirectives.h"

#include "JLM/JLMOps.h"

using namespace mlir;
using namespace jlm;

/**
 * Memory operators
 */

/**
 * Load
 */
// TODO: Either eliminate the function or provide useful verification
//       Used to check pointer type but pointers have become opaque
LogicalResult jlm::Load::verify() {
  return LogicalResult::success();
}

/*
* Store
*/
// TODO: Either eliminate the function or provide useful verification
//       Used to check pointer type but pointers have become opaque
LogicalResult jlm::Store::verify() {
  return LogicalResult::success();
}

/**
 * Alloca
 */
// TODO: Either eliminate the function or provide useful verification
//       Used to check pointer type but pointers have become opaque
LogicalResult jlm::Alloca::verify() {
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
