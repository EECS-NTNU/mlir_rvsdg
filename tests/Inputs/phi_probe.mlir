// As memset_probe.mlir: not collected as a test, run by the availability check
// in lit.cfg.py. What is being decided is not whether rvsdg.phiResult is
// registered, because it is registered either way, but whether it accepts the
// function-type references that a lambda node produces. A build that types
// those operands as lambdaRef rejects this file, so the file is the probe.
rvsdg.omegaNode (): {
  %lambda1, %lambda2 = rvsdg.phiNode ():
      (%f_ref: (f32) -> f32, %g_ref: () -> i32): {
    %lambda1 = rvsdg.lambdaNode {linkage = "external_linkage", sym_name = "f"} (f32) -> f32 ():
        (%test: f32): {
      %0 = arith.constant 1.0 : f32
      rvsdg.lambdaResult (%0 : f32)
    }
    %lambda2 = rvsdg.lambdaNode {linkage = "external_linkage", sym_name = "g"} () -> i32 ():
        (): {
      %1 = arith.constant 1 : i32
      rvsdg.lambdaResult (%1 : i32)
    }
    rvsdg.phiResult (%lambda1: (f32) -> f32, %lambda2: () -> i32)
  } -> (f32) -> f32, () -> i32
  %r:2 = rvsdg.omegaResult {exportNames = ["f", "g"]} (%lambda1: (f32) -> f32, %lambda2: () -> i32) : (f32) -> f32, () -> i32
}
