// RUN: %rvsdg-opt --split-input-file --verify-diagnostics %s
// REQUIRES: jlm-memset

// A volatile fill has externally visible side effects, so it threads an I/O
// state: a missing I/O state would let the fill be reordered past I/O it
// should be ordered with.
%p = "llvm.mlir.null" () : () -> !llvm.ptr
%v = arith.constant 0 : i8
%l = arith.constant 4 : i32
%mem = "jlm.undef"() : () -> !rvsdg.memState
// expected-error @+1 {{volatile memset requires an input and an output I/O state}}
"jlm.memset"(%p, %v, %l, %mem) <{isVolatile = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 0, 1>}> : (!llvm.ptr, i8, i32, !rvsdg.memState) -> (!rvsdg.memState)

// -----

// The other direction: a fill that does not claim to be volatile must not
// carry an I/O state, which would assert a side effect the op disclaims.
%p = "llvm.mlir.null" () : () -> !llvm.ptr
%v = arith.constant 0 : i8
%l = arith.constant 4 : i32
%io = "jlm.undef"() : () -> !rvsdg.ioState
// expected-error @+1 {{non-volatile memset must not have an input or output I/O state}}
"jlm.memset"(%p, %v, %l, %io) <{isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!llvm.ptr, i8, i32, !rvsdg.ioState) -> (!rvsdg.ioState)
