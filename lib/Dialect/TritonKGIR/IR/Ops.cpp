//===- Ops.cpp - TritonKGIR Operation Implementations -------*- C++ -*-===//
//
// Part of the Triton project.
//
// Stub implementation to unblock module build. Will be replaced by the
// assigned agent with full operation builders, verifiers, folders, and
// canonicalization patterns for KGIR operations.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonKGIR/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::kgir;

// Generated operation class implementations from TritonKGIROps.td.
#define GET_OP_CLASSES
#include "triton/Dialect/TritonKGIR/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// KernelLaunchOp::verify
//===----------------------------------------------------------------------===//

LogicalResult KernelLaunchOp::verify() {
  // grid_dims must have exactly 3 elements (x, y, z).
  auto gridDims = getGridDims();
  if (gridDims.size() != 3)
    return emitOpError("'grid_dims' must have exactly 3 elements, got ")
           << gridDims.size();

  // grid_dims elements must be positive.
  for (size_t i = 0; i < gridDims.size(); ++i) {
    if (gridDims[i] <= 0)
      return emitOpError("'grid_dims' element at index ")
             << i << " must be positive, got " << gridDims[i];
  }

  // num_warps must be positive.
  if (getNumWarps() <= 0)
    return emitOpError("'num_warps' must be positive, got ") << getNumWarps();

  // shared_memory_bytes must be non-negative.
  if (getSharedMemoryBytes() < 0)
    return emitOpError("'shared_memory_bytes' must be non-negative, got ")
           << getSharedMemoryBytes();

  // register_pressure must be non-negative.
  if (getRegisterPressure() < 0)
    return emitOpError("'register_pressure' must be non-negative, got ")
           << getRegisterPressure();

  // node_id must be non-negative.
  if (getNodeId() < 0)
    return emitOpError("'node_id' must be non-negative, got ") << getNodeId();

  return success();
}

//===----------------------------------------------------------------------===//
// FusedKernelOp::verify
//===----------------------------------------------------------------------===//

LogicalResult FusedKernelOp::verify() {
  // fused_node_ids must have at least 2 elements (need >=2 kernels to fuse).
  auto fusedIds = getFusedNodeIds();
  if (fusedIds.size() < 2)
    return emitOpError("'fused_node_ids' must have at least 2 elements, got ")
           << fusedIds.size();

  // fusion_type must be "producer_consumer" or "sibling".
  auto fusionType = getFusionType();
  if (fusionType != "producer_consumer" && fusionType != "sibling")
    return emitOpError("'fusion_type' must be \"producer_consumer\" or "
                       "\"sibling\", got \"")
           << fusionType << "\"";

  // combined_grid_dims must have exactly 3 elements (x, y, z).
  auto gridDims = getCombinedGridDims();
  if (gridDims.size() != 3)
    return emitOpError(
               "'combined_grid_dims' must have exactly 3 elements, got ")
           << gridDims.size();

  // combined_grid_dims elements must be positive.
  for (size_t i = 0; i < gridDims.size(); ++i) {
    if (gridDims[i] <= 0)
      return emitOpError("'combined_grid_dims' element at index ")
             << i << " must be positive, got " << gridDims[i];
  }

  // combined_shared_memory_bytes must be non-negative.
  if (getCombinedSharedMemoryBytes() < 0)
    return emitOpError(
               "'combined_shared_memory_bytes' must be non-negative, got ")
           << getCombinedSharedMemoryBytes();

  // combined_register_pressure must be non-negative.
  if (getCombinedRegisterPressure() < 0)
    return emitOpError(
               "'combined_register_pressure' must be non-negative, got ")
           << getCombinedRegisterPressure();

  // node_id must be non-negative.
  if (getNodeId() < 0)
    return emitOpError("'node_id' must be non-negative, got ") << getNodeId();

  return success();
}

//===----------------------------------------------------------------------===//
// GraphOp::verify
//===----------------------------------------------------------------------===//

LogicalResult GraphOp::verify() {
  // dispatch_mode must be "performance", "cost", or "balanced".
  auto mode = getDispatchMode();
  if (mode != "performance" && mode != "cost" && mode != "balanced")
    return emitOpError("'dispatch_mode' must be \"performance\", \"cost\", or "
                       "\"balanced\", got \"")
           << mode << "\"";

  // num_nodes must be non-negative.
  if (getNumNodes() < 0)
    return emitOpError("'num_nodes' must be non-negative, got ")
           << getNumNodes();

  // num_edges must be non-negative.
  if (getNumEdges() < 0)
    return emitOpError("'num_edges' must be non-negative, got ")
           << getNumEdges();

  // If iteration_count is present, it must be non-negative and <= 20.
  if (auto iterCount = getIterationCount()) {
    if (*iterCount < 0)
      return emitOpError("'iteration_count' must be non-negative, got ")
             << *iterCount;
  }

  return success();
}
