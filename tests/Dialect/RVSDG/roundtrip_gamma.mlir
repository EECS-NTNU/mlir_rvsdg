// RUN: %rvsdg-opt --verify-roundtrip %s | %FileCheck %s
//
// A gamma node is an n-way branch. Its predicate is a control value, so each
// region takes an extra leading argument for the match content alongside the
// gamma's own inputs, and every region's terminator must produce the node's
// result types.

rvsdg.omegaNode (): {
  %select_val = arith.constant 1 : i32
  // CHECK: rvsdg.match
  // CHECK-SAME: matchRule<0, 1 -> 1>
  // CHECK-SAME: matchRule<default -> 0>
  %predicate = rvsdg.match (%select_val : i32) [#rvsdg.matchRule<0, 1 -> 1>, #rvsdg.matchRule<default -> 0>] -> !rvsdg.ctrl<2>

  %some_variable = arith.constant 14.0 : f32
  // CHECK: rvsdg.gammaNode
  // CHECK-COUNT-2: rvsdg.gammaResult
  %gammaResult0, %gammaResult1 = rvsdg.gammaNode (%predicate : !rvsdg.ctrl<2>) (%some_variable : f32) : [
    (%matchContent : none, %input : f32) : {
      %c = arith.constant 1.0 : f32
      rvsdg.gammaResult (%c : f32, %input : f32)
    },
    (%matchContent : none, %input : f32) : {
      %c = arith.constant 2.0 : f32
      rvsdg.gammaResult (%c : f32, %input : f32)
    }
  ] -> f32, f32
  %r:2 = rvsdg.omegaResult {exportNames = ["out0", "out1"]} (%gammaResult0 : f32, %gammaResult1 : f32) : f32, f32
}
