// RUN: %rvsdg-opt --verify-roundtrip %s | %FileCheck %s
//
// A theta node is a tail-controlled loop: its first region result is the
// predicate deciding whether another iteration runs, and the rest feed either
// the next iteration's arguments or the node's outputs. Inputs, region
// arguments and outputs therefore have to agree one to one.

rvsdg.omegaNode (): {
  %test0 = arith.constant 1.0 : f32
  %test1 = arith.constant 1.0 : f64

  // CHECK: rvsdg.thetaNode
  // CHECK: rvsdg.thetaResult
  %res0, %res1 = rvsdg.thetaNode (%test0 : f32, %test1 : f64) :
      (%iter0 : f32, %iter1 : f64) : {
    %predicate = rvsdg.constantCtrl 0 : !rvsdg.ctrl<2>
    rvsdg.thetaResult (%predicate) : (%iter0 : f32, %iter1 : f64)
  } -> f32, f64

  %r = rvsdg.omegaResult {exportNames = ["res0"]} (%res0 : f32) : f32
}
