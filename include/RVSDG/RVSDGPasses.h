#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

#include "RVSDG/RVSDGDialect.h"
#include "RVSDG/RVSDGOps.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <polygeist/Dialect.h>

#define GEN_PASS_DECL
#include "RVSDG/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "RVSDG/Passes.h.inc"