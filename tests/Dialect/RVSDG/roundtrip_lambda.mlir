// RUN: %rvsdg-opt --verify-roundtrip %s | %FileCheck %s
//
// A lambda node models a function. Its result is the function's type, so the
// signature is printed directly rather than boxed inside a lambda reference.
// Region arguments are the parameters followed by any captured context values.

rvsdg.omegaNode (): {
    %ctx0 = arith.constant 4 : i32
    %ctx1 = arith.constant 5.0 : f32

    // CHECK: rvsdg.lambdaNode
    // CHECK-SAME: (i32, i32) -> f32
    // CHECK: rvsdg.lambdaResult
    %l = rvsdg.lambdaNode {linkage = "external_linkage", sym_name = "scaled"} (i32, i32) -> f32 (%ctx0 : i32, %ctx1 : f32) :
        (%param : i32, %ctx0 : i32, %ctx1 : f32) : {
        %0 = arith.muli %ctx0, %param : i32
        %1 = arith.sitofp %0 : i32 to f32
        %2 = arith.addf %1, %ctx1 : f32
        rvsdg.lambdaResult (%2 : f32)
    }

    // CHECK: jlm.call
    // CHECK-SAME: (i32, i32) -> f32
    %param = arith.constant 100 : i32
    %io0 = "jlm.undef"() : () -> !rvsdg.ioState
    %mem0 = "jlm.undef"() : () -> !rvsdg.memState
    %res, %io1, %mem1 = "jlm.call"(%l, %param, %io0, %mem0) : ((i32, i32) -> f32, i32, !rvsdg.ioState, !rvsdg.memState) -> (f32, !rvsdg.ioState, !rvsdg.memState)
    %r = rvsdg.omegaResult {exportNames = ["main"]} (%res : f32) : f32
}
