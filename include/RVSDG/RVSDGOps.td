#ifndef RVSDG_OPS
#define RVSDG_OPS

include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

include "RVSDG/RVSDGDialect.td"
include "RVSDG/RVSDGTypes.td"
include "RVSDG/RVSDGAttrs.td"
include "RVSDG/RVSDGInterfaces.td"

include "mlir/Dialect/LLVMIR/LLVMTypes.td"

class RVSDG_Op<string mnemonic, list<Trait> traits = []> :
    Op<RVSDG_Dialect, mnemonic, traits>;

class RVSDG_StructuralNode<string mnemonic, list<Trait> traits = []> : 
    RVSDG_Op<mnemonic, traits # [IsolatedFromAbove, RVSDG_StructuralNodeInterface]>;

/**
*  Gamma Node
*/
def RVSDG_GammaNode: RVSDG_StructuralNode<"gammaNode"> {
    let summary = "RVSDG gamma node which models conditional execution";
    let description = [{
        RVSDG gamma node. Models conditional execution such as if and switch statements.
        The node takes a predicate as its first input. The value of the predicate is used
        as the index of the region to be executed. Any other inputs are passed as arguments
        to all regions of the gamma node. The results of all regions are mapped to the
        outputs of the gamma node.
    }];
    let arguments = (ins
        RVSDG_Ctrl:$predicate,
        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let regions = (region
        VariadicRegion<SizedRegion<1>>:$regions
    );

    let assemblyFormat = "attr-dict `(` $predicate `:` type($predicate) `)` custom<TypedParamList>($inputs, type($inputs)) `:` custom<RVSDGRegions>($regions) `->` type(results)";
    let hasVerifier = 1;
}

def RVSDG_GammaResult: RVSDG_Op<"gammaResult", [
        HasParent<"mlir::rvsdg::GammaNode">,
        Terminator
    ]> {
    let summary = "Gamma node results";
    let description = [{
        Gamma node terminator. Maps operands to gamma node
        output.
    }];

    let arguments = (ins
        Variadic<AnyType>:$outputValues
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($outputValues, type($outputValues))";
    let hasVerifier = 1;
}


def RVSDG_OmegaResult: RVSDG_Op<"omegaResult", [
        HasParent<"mlir::rvsdg::OmegaNode">,
        Pure,
        Terminator
    ]> {
    let summary = "Omega node results";
    let description = [{
        Omega node terminator. Maps operands to items exported
        from the omega node.
    }];

    let arguments = (ins
        Variadic<AnyType>:$outputValues
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($outputValues, type($outputValues))";
}

def RVSDG_OmegaArgument: RVSDG_Op<"omegaArgument", [
        HasParent<"mlir::rvsdg::OmegaNode">,
        Pure,
    ]> {
    let summary = "Omega node argument";
    let description = [{
        Represents a single argument to an Omega Node. 
        This is a reference to an external value
    }];

    let arguments = (ins
        TypeAttr:$valueType,
        StrAttr:$linkage,
        StrAttr:$name
    );

    let results = (outs
        AnyTypeOf<[LLVMPointerType, FunctionType]>:$importedValue
    );

    let hasVerifier = 1;
}

/**
*  Omega Node
*/
def RVSDG_OmegaNode: RVSDG_StructuralNode<"omegaNode", [
    SingleBlockImplicitTerminator<"OmegaResult">
]> {
    let summary = "RVSDG Omega node. Top level node in RVSDG";
    let description = [{
        RVSDG omega node. This is the top level node in RVSDG and models a translation unit.
        Region arguments for this node are values that are imported from external sources.
        Region results from this node are exports that can be used in other translation units.
    }];
    let results = (outs);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "attr-dict custom<RVSDGRegion>($body)";
}

/**
*  Lambda Node
*/

def RVSDG_LambdaNode: RVSDG_StructuralNode<"lambdaNode", [DeclareOpInterfaceMethods<CallableOpInterface>]> {

    let summary = "RVSDG lambda node. Models functions and methods.";
    let description = [{
        RVSG lambda node. Models a function or method definition. Inputs to this node are context
        values needed at call-time. The output from this node is a reference to the defined function.
        The region of this node is the body of the function. Region arguments are function parameters
        followed by context values. Region results are the return values of the function.
    }];

    let arguments = (ins
        Variadic<AnyType>: $contextValues
    );

    let results = (outs
        FunctionType: $reference
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let assemblyFormat = "attr-dict type($reference) custom<TypedParamList>($contextValues, type($contextValues)) `:` custom<RVSDGRegion>($body)";

    let hasVerifier = 1;
}

def RVSDG_LambdaResult: RVSDG_Op<"lambdaResult", [
        HasParent<"mlir::rvsdg::LambdaNode">,
        Terminator,
        Pure
    ]> {
    let summary = "Lambda node results";
    let description = [{
        Lambda node terminator. Maps operands to outputs of apply
        node.
    }];

    let arguments = (ins
        Variadic<AnyType>:$outputValues
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($outputValues, type($outputValues))";

    let hasVerifier = 1;
}

def RVSDG_LambdaEntryMemoryStateSplitOperation: RVSDG_Op<"lambdaEntryMemoryStateSplitOperation", [
        HasParent<"mlir::rvsdg::LambdaNode">,
        Pure
    ]> {
    let summary = "Lambda entry memory splitter";
    let description = [{
        Splits incoming memory state into multiple states.
    }];

    let arguments = (ins
        RVSDG_MemState:$input
    );

    let results = (outs
        Variadic<RVSDG_MemState>:$outputValues
    );

//    let assemblyFormat = "attr-dict `(` $input `)` `:`custom<TypedParamList>($outputValues, type($outputValues))";
}

def RVSDG_LambdaExitMemoryStateMergeOperation: RVSDG_Op<"lambdaExitMemoryStateMergeOperation", [
        HasParent<"mlir::rvsdg::LambdaNode">,
        Pure
    ]> {
    let summary = "Lambda exit memory merger";
    let description = [{
        Merges memory state into a single outgoing memory states.
    }];

    let arguments = (ins
        Variadic<RVSDG_MemState>:$inputStates
    );

    let results = (outs
        RVSDG_MemState:$output
    );

    let assemblyFormat = "attr-dict `(` $inputStates `)` `:` type($output)";
}

def RVSDG_CallExitMemoryStateSplit: RVSDG_Op<"callExitMemoryStateSplitOperation", [
        Pure
    ]> {
    let summary = "Lambda entry memory splitter";
    let description = [{
        Splits incoming memory state into multiple states.
    }];

    let arguments = (ins
        RVSDG_MemState:$input
    );

    let results = (outs
        Variadic<RVSDG_MemState>:$outputValues
    );

//    let assemblyFormat = "attr-dict `(` $input `)` `:`custom<TypedParamList>($outputValues, type($outputValues))";
}

def RVSDG_CallEntryMemoryStateMerge: RVSDG_Op<"callEntryMemoryStateMerge", [
        Pure
    ]> {
    let summary = "Call memory merger";
    let description = [{
        Merges memory state into a single outgoing memory states.
    }];

    let arguments = (ins
        Variadic<RVSDG_MemState>:$inputStates
    );

    let results = (outs
        RVSDG_MemState:$output
    );

    let assemblyFormat = "attr-dict `(` $inputStates `)` `:` type($output)";
}


/*
*  Theta Node
*/

def RVSDG_ThetaNode: RVSDG_StructuralNode<"thetaNode", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "RVSDG theta node. Models a tail-controlled loop";
    let description = [{
        RVSDG theta node. Models a tail-controlled loop. First region result
        is a predicate that decides whether another iteration should occur.
        Other region results are mapped to region arguments for the next 
        iteration, or to the node outputs when the loop terminates.
    }];

    let arguments = (ins
        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($inputs, type($inputs)) `:` custom<RVSDGRegion>($body) `->` type(results)";
    let hasVerifier = 1;
}

def RVSDG_ThetaResult: RVSDG_Op<"thetaResult", [
    HasParent<"mlir::rvsdg::ThetaNode">,
    Terminator,
    Pure
]>{
    let summary = "Theta node results";
    let description = [{
        Theta node results. First operand is a predicate that decides whether the iteration
        should continue. Other operands map to the arguments of the next iteration if the
        iteration continues, or the outputs of the theta node otherwise.
    }];

    let arguments = (ins
        RVSDG_Ctrl<2>:$predicate,
        Variadic<AnyType>:$outputValues
    );

    let assemblyFormat = "attr-dict `(` $predicate `)` `:` custom<TypedParamList>($outputValues, type($outputValues))";
    let hasVerifier = 1;
}

/*
*  Phi Node
*/
def RVSDG_PhiNode: RVSDG_StructuralNode<"phiNode"> {
    let summary = "RVSDG phi node. Models an environment with mutually recursive functions";
    let description = [{
        RVSDG phi node. Models an environment with mutually recursive functions. Has a single region
        which can contain lambda and delta nodes. Inputs to this node capture context values that are
        needed by the contained nodes. Outputs are references to the contained mutually recursive functions.
        The region arguments consist of the context values passed through the inputs, and references
        to the enclosed lambda nodes. The region results map to the region arguments which are lambda
        references, and to the outputs of the phi node.
    }];

    let arguments = (ins
        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let regions = (region
        SizedRegion<1>:$body
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($inputs, type($inputs)) `:` custom<RVSDGRegion>($body) `->` type($outputs)";
    let hasVerifier = 1;
}

def RVSDG_PhiResult: RVSDG_Op<"phiResult", [
    HasParent<"mlir::rvsdg::PhiNode">,
    Terminator,
    Pure
]>{
    let summary = "Phi node results";
    let description = [{
        Phi node results. Maps operands to the region arguments which are lambda
        references, and to the outputs of the phi node.
    }];

    let arguments = (ins
        Variadic<RVSDG_LambdaRef>:$results
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($results, type($results))";
    let hasVerifier = 1;
}

/*
*  Delta Node
*/
def RVSDG_DeltaNode: RVSDG_StructuralNode<"deltaNode", [Pure]>{
    let summary = "RVSDG delta node. Used to model global variables";

    let description = [{
        RVSDG delta node. Used to model global variables. Inputs to the node model
        context values that the global variable depends on. The output of the node
        is a single pointer that points to the modelled variable. Region arguments
        are mapped one to one from the node inputs. The region result is the value
        of the modelled variable.
    }];

    let arguments = (ins
        Variadic<AnyType>:$inputs, 
        StrAttr:$name,
        StrAttr:$linkage,
        StrAttr:$section,
        BoolAttr:$constant
    );

    let results = (outs
        LLVMPointerType: $output
    );

    let regions = (region
        SizedRegion<1>: $body
    );

    let assemblyFormat = "attr-dict custom<TypedParamList>($inputs, type($inputs)) `:` custom<RVSDGRegion>($body) `->` type($output)";
    let hasVerifier = 1;
}

def RVSDG_DeltaResult: RVSDG_Op<"deltaResult", [HasParent<"mlir::rvsdg::DeltaNode">, Terminator, Pure]> {
    let summary = "Delta node results";
    let description = [{
        Delta node results. Takes the value of the modelled global variable as input.
    }];

    let arguments = (ins
        AnyType:$input
    );

    let assemblyFormat = "attr-dict `(` $input `:` type($input) `)`";
    let hasVerifier = 1;
}

/*
*  Control type ops
*/

def RVSDG_ConstantCtrl: RVSDG_Op<"constantCtrl", [Pure]> {
    let summary = "Materialize constant control value from integer constant";
    let description = [{
        Creates a constant control value from a positive integer constant.
    }];

    let arguments = (ins
        UI64Attr:$value
    );

    let results = (outs
        RVSDG_Ctrl:$output
    );

    let assemblyFormat = "attr-dict $value `:` type($output)";
}

def RVSDG_Match: RVSDG_Op<"match", [Pure]> {
    let summary = "Match operator. Lookup table that converts integers into control values";
    let description = [{
        Match operator. Lookup table that converts integers into control values.
        The first input is the integer to be matched. The second input is the
        default control value. The remaining inputs are pairs of integers and
        control values. The first integer of each pair is matched against the
        first input. If the match succeeds, the corresponding control value is
        returned. If no match is found, the default control value is returned.
    }];

    let arguments = (ins
        AnyInteger:$input,
        TypedArrayAttrBase<RVSDG_MatchRuleAttr, "Array of mapping rules">:$mapping
    );

    let results = (outs
        RVSDG_Ctrl:$output
    );

    let assemblyFormat = "attr-dict `(` $input `:` type($input) `)` $mapping `->` type($output)";
    let hasVerifier = 1;
}

/*
*  State management ops
*/

def RVSDG_MemStateMerge: RVSDG_Op<"memStateMerge", [Pure]> {
    let summary = "Merge two memory states";
    let description = [{
        Merge two memory states.
    }];

    let arguments = (ins
        Variadic<RVSDG_MemState>:$inputStates
    );

    let results = (outs
        RVSDG_MemState:$output
    );

    let assemblyFormat = "attr-dict `(` $inputStates `)` `:` type($output)";
}
#endif // RVSDG_OPS