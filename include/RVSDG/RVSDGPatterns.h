#pragma once

#include <memory>
#include <mlir/Pattern/Pattern.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <polygeist/Dialect.h>
#include <RVSDG/RVSDGDialect.h>
#include <RVSDG/RVSDGOps.h>

#define GEN_PASS_DECL
#include <RVSDG/Patterns.h.inc>

// Declare pass creation functions
namespace mlir
{
namespace rvsdg
{
} // namespace rvsdg
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "RVSDG/Patterns.h.inc"
