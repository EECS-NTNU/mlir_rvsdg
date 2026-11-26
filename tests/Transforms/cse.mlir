// RUN: %rvsdg-opt --cse %s | %FileCheck %s
//
// CSE collapses two identical structural nodes into one. The pair here is pure
// and does collapse: what survives is a single thetaNode, and its two operands
// are the same value because the duplicated constant merged with it.
//
// The second pair carries a memory state edge, and CSE leaves both nodes in
// place. The test asserts that rather than assuming it, so a change to how the
// pass treats state edges shows up here instead of silently.

rvsdg.omegaNode (): {
  %x = arith.constant 10 : i32
  %y = arith.constant 10 : i32

  // CHECK-LABEL: rvsdg.omegaNode
  // CHECK-COUNT-1: arith.constant
  // CHECK-COUNT-1: rvsdg.thetaNode
  // CHECK-NOT: rvsdg.thetaNode
  // CHECK: jlm.undef
  // CHECK-COUNT-2: rvsdg.thetaNode
  // CHECK: rvsdg.omegaResult
  %t1:2 = rvsdg.thetaNode (%x : i32, %y : i32) :
      (%a : i32, %b : i32) : {
    %predicate = rvsdg.constantCtrl 0 : !rvsdg.ctrl<2>
    rvsdg.thetaResult (%predicate) : (%a : i32, %b : i32)
  } -> i32, i32

  %t2:2 = rvsdg.thetaNode (%x : i32, %y : i32) :
      (%a : i32, %b : i32) : {
    %predicate = rvsdg.constantCtrl 0 : !rvsdg.ctrl<2>
    rvsdg.thetaResult (%predicate) : (%a : i32, %b : i32)
  } -> i32, i32

  %root_mem_state = "jlm.undef"() : () -> !rvsdg.memState
  %t3:3 = rvsdg.thetaNode (%x : i32, %y : i32, %root_mem_state : !rvsdg.memState) :
      (%a : i32, %b : i32, %m : !rvsdg.memState) : {
    %predicate = rvsdg.constantCtrl 0 : !rvsdg.ctrl<2>
    rvsdg.thetaResult (%predicate) : (%a : i32, %b : i32, %m : !rvsdg.memState)
  } -> i32, i32, !rvsdg.memState

  %t4:3 = rvsdg.thetaNode (%x : i32, %y : i32, %root_mem_state : !rvsdg.memState) :
      (%a : i32, %b : i32, %m : !rvsdg.memState) : {
    %predicate = rvsdg.constantCtrl 0 : !rvsdg.ctrl<2>
    rvsdg.thetaResult (%predicate) : (%a : i32, %b : i32, %m : !rvsdg.memState)
  } -> i32, i32, !rvsdg.memState

  %r:2 = rvsdg.omegaResult {exportNames = ["u", "v"]} (%t2#0 : i32, %t4#1 : i32) : i32, i32
}
