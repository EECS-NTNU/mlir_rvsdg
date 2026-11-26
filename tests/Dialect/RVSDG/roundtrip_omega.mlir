// RUN: %rvsdg-opt --verify-roundtrip %s | %FileCheck %s
//
// The omega node is the translation unit. Region arguments are values imported
// from outside the unit, and the terminator names what it exports, so
// omegaResult carries the types of its operands alongside its own results.

rvsdg.omegaNode (): {
  // CHECK-LABEL: rvsdg.omegaNode
  // CHECK: rvsdg.omegaArgument
  %imported = "rvsdg.omegaArgument"() <{linkage = "external_linkage", name = "ext", valueType = (!llvm.ptr, !rvsdg.ioState, !rvsdg.memState) -> (!rvsdg.ioState, !rvsdg.memState)}> : () -> ((!llvm.ptr, !rvsdg.ioState, !rvsdg.memState) -> (!rvsdg.ioState, !rvsdg.memState))
  %exported = arith.constant 1.0 : f32
  // CHECK: rvsdg.omegaResult
  // CHECK-SAME: exportNames = ["exported"]
  %r = rvsdg.omegaResult {exportNames = ["exported"]} (%exported : f32) : f32
}
