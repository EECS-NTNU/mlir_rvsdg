// RUN: %rvsdg-opt --verify-roundtrip %s | %FileCheck %s
// REQUIRES: phi-result-typed-by-reference
//
// A phi node models a set of mutually recursive definitions. It produces a
// reference per contained function, and since a lambda's reference is now its
// function type, the phi's outputs and its leading region arguments are typed
// that way too — a phi region argument list is the output references first,
// then any context values passed through the inputs.

rvsdg.omegaNode (): {
    // CHECK: rvsdg.phiNode
    // CHECK: rvsdg.lambdaNode
    // CHECK: rvsdg.phiResult
    %lambda1, %lambda2 = rvsdg.phiNode () :
        (%f_ref : (f32) -> f32, %g_ref : () -> i32) : {
        %lambda1 = rvsdg.lambdaNode {linkage = "external_linkage", sym_name = "f"} (f32) -> f32 () :
            (%test : f32) : {
            %0 = arith.constant 1.0 : f32
            rvsdg.lambdaResult (%0 : f32)
        }
        %lambda2 = rvsdg.lambdaNode {linkage = "external_linkage", sym_name = "g"} () -> i32 () :
            () : {
            %1 = arith.constant 1 : i32
            rvsdg.lambdaResult (%1 : i32)
        }
        rvsdg.phiResult (%lambda1 : (f32) -> f32, %lambda2 : () -> i32)
    } -> (f32) -> f32, () -> i32

    %r:2 = rvsdg.omegaResult {exportNames = ["f", "g"]} (%lambda1 : (f32) -> f32, %lambda2 : () -> i32) : (f32) -> f32, () -> i32
}
