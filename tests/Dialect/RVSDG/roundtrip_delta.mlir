// RUN: %rvsdg-opt --verify-roundtrip %s | %FileCheck %s
//
// A delta node models a global: it produces a pointer, and the region result
// is the value that pointer addresses. Note !llvm.ptr carries no element type
// of its own — the pointee is whatever deltaResult yields.

rvsdg.omegaNode (): {
    // CHECK: rvsdg.deltaNode
    // CHECK-SAME: name = "global_i16"
    %10 = rvsdg.deltaNode {constant = false, linkage = "external_linkage", name = "global_i16", section = ""} () :
        () : {
        %11 = arith.constant 10 : i16
        rvsdg.deltaResult (%11 : i16)
    } -> !llvm.ptr

    %ctx = arith.constant 1 : i32
    // CHECK: rvsdg.deltaNode
    // CHECK-SAME: name = "global_i32"
    // CHECK: rvsdg.deltaResult
    %11 = rvsdg.deltaNode {constant = true, linkage = "internal_linkage", name = "global_i32", section = ""} (%ctx : i32) :
        (%captured : i32) : {
        %12 = arith.constant 20 : i32
        rvsdg.deltaResult (%12 : i32)
    } -> !llvm.ptr

    %r:2 = rvsdg.omegaResult {exportNames = ["global_i16", "global_i32"]} (%10 : !llvm.ptr, %11 : !llvm.ptr) : !llvm.ptr, !llvm.ptr
}
