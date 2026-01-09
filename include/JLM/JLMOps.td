include "mlir/IR/BuiltinTypes.td"
include "RVSDG/RVSDGTypes.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

include "JLM/JLMDialect.td"
include "JLM/JLMTypes.td"


class JLM_Op<string mnemonic, list<Trait> traits = []> :
    Op<JLM_Dialect, mnemonic, traits>;


def JLM_Load: JLM_Op<"load", []> {
    let summary = "Load from memory";
    let description = [{
        Load from memory.
    }];

    let arguments = (ins
        LLVMPointerType:$pointer,
        UI32Attr:$alignment,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        AnyType:$output,
        Variadic<RVSDG_MemState>:$outputMemStates
    );

    let assemblyFormat = "attr-dict $pointer `:` type($pointer) `(` $inputMemStates `)` `->` type($output) `,` type($outputMemStates)";
    let hasVerifier = 1;
}

def JLM_Store: JLM_Op<"store"> {
    let summary = "Store to memory";
    let description = [{
        Store to memory.
    }];

    let arguments = (ins
        LLVMPointerType:$pointer,
        AnyType:$value,
        UI32Attr:$alignment,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        Variadic<RVSDG_MemState>:$outputMemStates
    );

    let assemblyFormat = "attr-dict `(` $pointer `:` type($pointer) `,` $value `:` type($value) `)` `(` $inputMemStates `)` `->` type($outputMemStates)";
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
        Variadic<RVSDG_MemState>:$inputMemStates,
        RVSDG_IOState:$inputIoState
    );

    let results = (outs
        Variadic<RVSDG_MemState>:$outputMemStates,
        RVSDG_IOState:$outputIoState
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

def JLM_Memcpy: JLM_Op<"memcpy"> {
    let summary = "Copy memory content from one location to another";
    let description = [{
        Copy memory content from one location to another. Same parameters as LLVM memcpy,
        but also interacts takes and produces memory state values for use with RVSDG.
    }];

    let arguments = (ins 
        LLVMPointerType:$dst,
        LLVMPointerType:$src,
        AnySignlessInteger:$len,
        AnyTypeOf<[I1, RVSDG_Ctrl<2>]>: $isVolatile,
        Variadic<RVSDG_MemState>:$inputMemStates
    );

    let results = (outs
        RVSDG_MemState:$outputMemState
    );

    let assemblyFormat = "attr-dict $dst `:` type($dst) `,` $src `:` type($src) `,` $len `:` type($len) `,` $isVolatile `:` type($isVolatile) (`,` $inputMemStates^)? `->` type($outputMemState)";
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
