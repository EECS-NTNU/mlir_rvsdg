#pragma once
#include <mlir/IR/Region.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

#include <JLM/JLMDialect.h>
#include <JLM/JLMTypes.h>
#include <RVSDG/RVSDGTypes.h>

#define GET_OP_CLASSES
#include <JLM/Ops.h.inc>