include "mlir/IR/PatternBase.td"

include "mlir/Dialect/MemRef/IR/MemRefOps.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"

include "mlir/Dialect/LLVMIR/LLVMOps.td"

include "RVSDG/RVSDGOps.td"
include "polygeist/PolygeistOps.td"

// // Pattern to remove memref.cast operation by replacing it with its input
// def : Pat<(MemRef_CastOp:$op $input), (replaceWithValue $input)>;

// // Pattern to convert LLVM.mlir.global to RVSDG deltanode
// def ConvertGlobalToDelta : NativeCodeCall<
//     "convertLLVMGlobalToRVSDGDelta($_builder, $0, $1, $2)">;

// def : Pat<(LLVM_GlobalOp:$op $sym_name, $type, $constant, $value, $linkage, $dso_local, $thread_local),
//           (ConvertGlobalToDelta $op, $sym_name, $value)>;

// def : Pat<(MemRef_AllocOp:$op $alignment, $dynamicSizes, $symbolOperands), (LLVM_AllocaOp $alignment, $input2, $input3)>;
// def : Pat<(MemRef_DeallocOp:$op $input), (replaceWithValue $input)>;
// def : Pat<(MemRef_LoadOp:$op $input), (replaceWithValue $input)>;
// def : Pat<(MemRef_StoreOp:$op $input), (replaceWithValue $input)>;

// def : Pat<(Pointer2MemrefOp:$op $input), (replaceWithValue $input)>;
