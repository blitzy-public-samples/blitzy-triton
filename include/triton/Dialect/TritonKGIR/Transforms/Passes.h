#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonKGIR/IR/Dialect.h"
#include <memory>

namespace mlir::triton::kgir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonKGIR/Transforms/Passes.h.inc"

} // namespace mlir::triton::kgir
