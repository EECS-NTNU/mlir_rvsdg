include "mlir/IR/BuiltinTypes.td"
include "RVSDG/RVSDGTypes.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

include "JLM/JLMDialect.td"
include "JLM/JLMTypes.td"


class JLM_Op<string mnemonic, list<Trait> traits = []> :
    Op<JLM_Dialect, mnemonic, traits>;


def JLM_Load: JLM_Op<"load", [AttrSizedOperandSegments, AttrSizedResultSegments]> {
    let summary = "Load from memory";
    let description = [{
        Load from memory.

        The `isVolatile` attribute is a flag indicating whether the load is volatile.
        A volatile load additionally threads an I/O state: it takes an input I/O state
        (`inputIoState`) and produces an output I/O state (`outputIoState`), modelling the
        externally-visible side effect. A non-volatile load carries neither I/O state. The
        remaining operands/results are the memory states.

        The verifier checks that `isVolatile` agrees with the presence of `inputIoState`
        and `outputIoState`.
    }];

    let arguments = (ins
        LLVMPointerType:$pointer,
        UI32Attr:$alignment,
        BoolAttr:$isVolatile,
        Optional<RVSDG_IOState>:$inputIoState,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        AnyType:$output,
        Optional<RVSDG_IOState>:$outputIoState,
        Variadic<RVSDG_MemState>:$outputMemStates
    );

    let hasVerifier = 1;
}

def JLM_Store: JLM_Op<"store", [AttrSizedOperandSegments, AttrSizedResultSegments]> {
    let summary = "Store to memory";
    let description = [{
        Store to memory.

        The `isVolatile` attribute is a flag indicating whether the store is volatile.
        A volatile store additionally threads an I/O state: it takes an input I/O state
        (`inputIoState`) and produces an output I/O state (`outputIoState`), modelling the
        externally-visible side effect. A non-volatile store carries neither I/O state. The
        remaining operands/results are the memory states.

        The verifier checks that `isVolatile` agrees with the presence of `inputIoState`
        and `outputIoState`.
    }];

    let arguments = (ins
        LLVMPointerType:$pointer,
        AnyType:$value,
        UI32Attr:$alignment,
        BoolAttr:$isVolatile,
        Optional<RVSDG_IOState>:$inputIoState,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        Optional<RVSDG_IOState>:$outputIoState,
        Variadic<RVSDG_MemState>:$outputMemStates
    );

    let hasVerifier = 1;
}

def JLM_Alloca: JLM_Op<"alloca"> {
    let summary = "Allocate memory on the stack";
    let description = [{
        Allocate memory on the stack.
    }];

    let arguments = (ins
        TypeAttr:$valueType,
        AnySignlessInteger:$size,
        UI32Attr:$alignment,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        LLVMPointerType:$output,
        RVSDG_MemState:$outputMemState
    );

    // let assemblyFormat = "attr-dict $size, $alignment `(` $inputMemStates `)` `->` type($output) `,` type($outputMemState)";
    let hasVerifier = 1;
}

def JLM_Call: JLM_Op<"call"> {
    let summary = "Call a function";
    let description = [{
        Call a function.
    }];

    let arguments = (ins
        FunctionType:$callee,
        Variadic<AnyType>:$args,
        RVSDG_IOState:$inputIoState,
        RVSDG_MemState:$inputMemState
    );

    let results = (outs
        Optional<AnyType>:$outputValue,
        RVSDG_IOState:$outputIoState,
        RVSDG_MemState:$outputMemState
    );

    // let assemblyFormat = "attr-dict $size, $alignment `(` $inputMemStates `)` `->` type($output) `,` type($outputMemState)";
    // let hasVerifier = 1;
}

def JLM_Free: JLM_Op<"free"> {
    let summary = "Free memory";
    let description = [{
        Free memory.
    }];

    let arguments = (ins
        LLVMPointerType:$input,
        RVSDG_IOState:$inputIoState,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        RVSDG_IOState:$outputIoState,
        Variadic<RVSDG_MemState>:$outputMemStates
    );
}

def JLM_Malloc: JLM_Op<"malloc"> {
    let summary = "Allocate heap memory";
    let description = [{
        Allocate heap memory.
    }];

    let arguments = (ins
        AnySignlessInteger:$size,
        RVSDG_IOState:$inputIoState
    );

    let results = (outs
        LLVMPointerType:$output,
        RVSDG_IOState:$outputIoState,
        RVSDG_MemState:$outputMemState
    );
}

def JLM_Memcpy: JLM_Op<"memcpy", [AttrSizedOperandSegments, AttrSizedResultSegments]> {
    let summary = "Copy memory content from one location to another";
    let description = [{
        Copy memory content from one location to another. Same parameters as LLVM memcpy,
        but also interacts takes and produces memory state values for use with RVSDG.

        The `isVolatile` operand is a 1-bit flag indicating whether the copy is volatile.
        A volatile copy additionally threads an I/O state: it takes an input I/O state
        (`inputIoState`) and produces an output I/O state (`outputIoState`), modelling the
        externally-visible side effect. A non-volatile copy carries neither I/O state. The
        remaining operands/results are the memory states.

        The verifier checks that `isVolatile` agrees with the presence of `inputIoState`
        and `outputIoState`.
    }];

    let arguments = (ins
        LLVMPointerType:$dst,
        LLVMPointerType:$src,
        AnySignlessInteger:$len,
        BoolAttr:$isVolatile,
        Optional<RVSDG_IOState>:$inputIoState,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        Optional<RVSDG_IOState>:$outputIoState,
        Variadic<RVSDG_MemState>:$outputMemStates
    );

    let hasVerifier = 1;
}

def JLM_CreateVarArgList: JLM_Op<"createVarargs"> {
    let summary = "Create a list of variadic arguments";
    let description = [{
        Create a list of variadic arguments from severeal SSA values.
    }];

    let arguments = (ins
        Variadic<AnyType>:$input
    );

    let results = (outs
        JLM_VarargList:$output
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($input, type($input)) `->` type($output)";
}

def JLM_Undef: JLM_Op<"undef"> {
    let summary = "Create an undefined value";
    let description = [{
        Create an undefined value.
    }];

    let results = (outs
        AnyType:$output
    );

    let assemblyFormat = "attr-dict `:` type($output)";
}

def JLM_ConstantDataArray: JLM_Op<"constantDataArray"> {
    let summary = "Constant data array operation";
    let description = [{
        Constant data array operation.
    }];

    let arguments = (ins
        Variadic<AnyType>:$input
    );

    let results = (outs
        LLVMArrayType:$output
    );

    let hasVerifier = 1;
}

def JLM_ConstantArray: JLM_Op<"constantArray"> {
    let summary = "Constant array operation";
    let description = [{
        Constant array operation.

        An array value built from its element values, which are the operands. Unlike
        `constantDataArray` the element values are not required to share a type, so this is
        the operation that carries an aggregate array, e.g. an array of structs whose
        elements are themselves aggregate values.
    }];

    let arguments = (ins
        Variadic<AnyType>:$input
    );

    let results = (outs
        LLVMArrayType:$output
    );
}

def JLM_ConstantStruct: JLM_Op<"constantStruct"> {
    let summary = "Constant struct operation";
    let description = [{
        Constant struct operation.
    }];

    let arguments = (ins
        Variadic<AnyType>:$input
    );

    let results = (outs
        AnyType:$output
    );
}

def JLM_IOBarrier: JLM_Op<"ioBarrier"> {
    let summary = "IO barrier operation";
    let description = [{
        IO barrier operation.
    }];

    let arguments = (ins
        AnyType:$input,
        RVSDG_IOState:$inputIoState
    );

    let results = (outs
        AnyType:$output
    );

    let hasVerifier = 1;
}

def JLM_MemoryHoistBarrier: JLM_Op<"memoryHoistBarrier"> {
    let summary = "Memory hoist barrier operation";
    let description = [{
        Sequentializes a memory operation, such as a load, store or memory copy, after an I/O
        state operation.

        The address is routed through the barrier together with an I/O state, so that the
        memory operation consuming the result is ordered after the producer of the I/O state.
        `dereferenceableSize` is the number of bytes the address is known to be dereferenceable
        over; 0 means unknown.

        %address, %ioState = ...
        %barrieredAddress = jlm.memoryHoistBarrier %address, %ioState
            {dereferenceableSize = 4 : ui64} : !llvm.ptr, !rvsdg.ioState -> !llvm.ptr
    }];

    let arguments = (ins
        LLVMPointerType:$address,
        RVSDG_IOState:$inputIoState,
        UI64Attr:$dereferenceableSize
    );

    let results = (outs
        LLVMPointerType:$output
    );

    let hasVerifier = 1;
}

def JLM_FuncToPtr: JLM_Op<"funcToPtr"> {
    let summary = "Cast a function reference to a pointer";
    let description = [{
        Interpret a reference to a function as a pointer.
    }];

    let arguments = (ins
        FunctionType:$input
    );

    let results = (outs
        LLVMPointerType:$output
    );
}

def JLM_PtrToFunc: JLM_Op<"ptrToFunc"> {
    let summary = "Cast a pointer to a function reference";
    let description = [{
        Interpret a pointer as a reference to a function of the given type.
    }];

    let arguments = (ins
        LLVMPointerType:$input
    );

    let results = (outs
        FunctionType:$output
    );
}
