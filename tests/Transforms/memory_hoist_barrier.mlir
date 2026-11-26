// RUN: %rvsdg-opt --verify-roundtrip %s | %FileCheck %s
//
// Input exercising jlm.memoryHoistBarrier. A non-volatile load/store address is
// routed through the barrier together with an I/O state, with the known
// dereferenceable size carried as a ui64 attribute.

// CHECK: rvsdg.lambdaNode
rvsdg.omegaNode (): {
  %0 = rvsdg.lambdaNode {linkage = "external_linkage", sym_name = "f"} (!llvm.ptr, !rvsdg.ioState, !rvsdg.memState) -> (!llvm.ptr, !rvsdg.ioState) () : (%arg0: !llvm.ptr, %arg1: !rvsdg.ioState, %arg2: !rvsdg.memState): {
    // CHECK: jlm.memoryHoistBarrier
    // CHECK-SAME: dereferenceableSize = 4
    %barriered = "jlm.memoryHoistBarrier"(%arg0, %arg1) <{dereferenceableSize = 4 : ui64}> : (!llvm.ptr, !rvsdg.ioState) -> !llvm.ptr
    %ptr, %memState = "jlm.load"(%barriered, %arg2) <{alignment = 4 : ui32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 1>, resultSegmentSizes = array<i32: 1, 0, 1>}> : (!llvm.ptr, !rvsdg.memState) -> (!llvm.ptr, !rvsdg.memState)
    rvsdg.lambdaResult(%ptr: !llvm.ptr, %arg1: !rvsdg.ioState)
  }
  // CHECK: rvsdg.omegaResult
  rvsdg.omegaResult {exportNames = ["f"]} (%0: (!llvm.ptr, !rvsdg.ioState, !rvsdg.memState) -> (!llvm.ptr, !rvsdg.ioState)) : (!llvm.ptr, !rvsdg.ioState, !rvsdg.memState) -> (!llvm.ptr, !rvsdg.ioState)
}
