// Probe file, not a test. The whole point is that this is not a valid
// program for the op; the question asked is only whether the driver knows the
// name at all, which it answers before it looks at the operands.
%0 = "jlm.memset"(): () -> i8
