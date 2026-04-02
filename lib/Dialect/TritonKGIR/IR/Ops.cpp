//===- Ops.cpp - TritonKGIR Operation Implementations -------*- C++ -*-===//
//
// Part of the Triton project.
//
// Operation builders, verifiers, folders, and canonicalization patterns for
// the 6 KGIR operations defined in TritonKGIROps.td. This file provides the
// hand-written C++ implementation glue that extends the TableGen-generated
// operation code.
//
// The KGIR (Kernel Graph IR) dialect models a directed acyclic graph (DAG)
// of kernel launches for graph-level cross-kernel optimization. Operations
// are structural/metadata — they record kernel launch information, dependency
// edges, cross-device transfers, fusion results, and graph-level container
// data. They do NOT perform tensor computation.
//
// Hand-written verifiers enforce invariants that cannot be expressed in
// TableGen alone:
//   - KernelLaunchOp::verify  — grid geometry, resource bounds, node identity
//   - FusedKernelOp::verify   — fusion provenance, strategy validity, resources
//   - GraphOp::verify         — dispatch mode, graph metadata, DAG acyclicity
//
// The DAG acyclicity check in GraphOp::verify uses Kahn's algorithm to
// detect cycles among kernel_launch and fused_kernel nodes connected by
// data_dep and anti_dep edges.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonKGIR/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::triton::kgir;

// Materialize all generated operation classes from TritonKGIROps.td TableGen
// definitions. This provides generated builders, printers, parsers, and the
// base verify() dispatch infrastructure for all 6 KGIR operations:
//   KernelLaunchOp, DataDepOp, AntiDepOp, TransferOp, FusedKernelOp, GraphOp
#define GET_OP_CLASSES
#include "triton/Dialect/TritonKGIR/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// KernelLaunchOp::verify
//===----------------------------------------------------------------------===//
//
// Verifies structural invariants for a single kernel launch node:
//   - node_id >= 0 (unique non-negative identifier within the graph)
//   - grid_dims has exactly 3 elements (x, y, z launch grid dimensions)
//   - All grid_dims elements are positive (non-degenerate grid)
//   - num_warps > 0 (at least one warp/wavefront)
//   - shared_memory_bytes >= 0 (resource consumption bound)
//   - register_pressure >= 0 (resource consumption bound)
//
LogicalResult KernelLaunchOp::verify() {
  // node_id must be non-negative. The node_id uniquely identifies this kernel
  // within the KGIR DAG for cross-referencing by edge operations.
  if (getNodeId() < 0)
    return emitOpError("node_id must be non-negative");

  // grid_dims must have exactly 3 elements representing the (x, y, z) launch
  // grid dimensions. This matches CUDA/HIP grid semantics.
  auto gridDims = getGridDims();
  if (gridDims.size() != 3)
    return emitOpError("grid_dims must have exactly 3 elements");

  // Each grid dimension must be positive to define a non-degenerate launch
  // grid. Zero or negative grid dimensions would result in no kernel execution
  // or undefined behavior on GPU hardware.
  for (size_t i = 0; i < gridDims.size(); ++i) {
    if (gridDims[i] <= 0)
      return emitOpError("grid_dims element at index ")
             << i << " must be positive, got " << gridDims[i];
  }

  // num_warps must be positive. Every kernel launch requires at least one
  // warp (NVIDIA) or wavefront (AMD) for execution.
  if (getNumWarps() <= 0)
    return emitOpError("num_warps must be positive");

  // shared_memory_bytes must be non-negative. A value of 0 indicates no
  // shared memory usage. The fusion analysis engine uses this to check
  // combined resource budgets against per-SM/CU hardware limits.
  if (getSharedMemoryBytes() < 0)
    return emitOpError("shared_memory_bytes must be non-negative");

  // register_pressure must be non-negative. This represents the estimated
  // register file consumption per thread and is used by the scheduler for
  // resource-aware SM/CU bin-packing decisions.
  if (getRegisterPressure() < 0)
    return emitOpError("register_pressure must be non-negative");

  return success();
}

//===----------------------------------------------------------------------===//
// FusedKernelOp::verify
//===----------------------------------------------------------------------===//
//
// Verifies structural invariants for a fused kernel node:
//   - fused_node_ids is non-empty (must record source kernel provenance)
//   - fusion_type is "producer_consumer" or "sibling"
//   - combined_grid_dims has exactly 3 elements (x, y, z)
//   - All combined_grid_dims elements are positive
//   - combined_shared_memory_bytes >= 0
//   - combined_register_pressure >= 0
//   - node_id >= 0
//
LogicalResult FusedKernelOp::verify() {
  // fused_node_ids (source_node_ids) must not be empty. This array records
  // the original kernel node IDs that were merged by the fusion analysis
  // engine, preserving provenance for debugging, rollback during closed-loop
  // feedback, and the code generation bridge which needs to merge kernel
  // bodies from the recorded source nodes.
  auto fusedIds = getFusedNodeIds();
  if (fusedIds.empty())
    return emitOpError("source_node_ids must not be empty");

  // A meaningful fusion requires at least 2 source kernels. A single-kernel
  // "fusion" is semantically equivalent to no fusion and indicates a logic
  // error in the fusion analysis engine.
  if (fusedIds.size() < 2)
    return emitOpError("fused_node_ids must have at least 2 elements, got ")
           << fusedIds.size();

  // fusion_type must be one of the two supported fusion strategies:
  //   "producer_consumer" — Kernel A writes a tensor that Kernel B reads
  //     with no other consumers; the intermediate is promoted from global
  //     memory to shared memory or registers.
  //   "sibling" — Independent kernels with compatible grid geometries are
  //     merged into a single launch with partitioned SM/CU allocation.
  auto fusionType = getFusionType();
  if (fusionType != "producer_consumer" && fusionType != "sibling")
    return emitOpError(
        "fusion_type must be 'producer_consumer' or 'sibling'");

  // combined_grid_dims must have exactly 3 elements for the unified (x, y, z)
  // grid geometry of the fused kernel.
  auto gridDims = getCombinedGridDims();
  if (gridDims.size() != 3)
    return emitOpError(
        "combined_grid_dims must have exactly 3 elements");

  // Each combined grid dimension must be positive.
  for (size_t i = 0; i < gridDims.size(); ++i) {
    if (gridDims[i] <= 0)
      return emitOpError("combined_grid_dims element at index ")
             << i << " must be positive, got " << gridDims[i];
  }

  // combined_shared_memory_bytes must be non-negative. This is the aggregate
  // shared memory consumption of the fused kernel, checked against per-SM/CU
  // hardware limits from the HardwareProfile during fusion legality analysis.
  if (getCombinedSharedMemoryBytes() < 0)
    return emitOpError("combined_shared_memory_bytes must be non-negative");

  // combined_register_pressure must be non-negative. This is the aggregate
  // register file consumption of the fused kernel.
  if (getCombinedRegisterPressure() < 0)
    return emitOpError("combined_register_pressure must be non-negative");

  // node_id must be non-negative. The fused kernel receives a new unique
  // node_id within the graph, distinct from the source kernel node_ids
  // recorded in fused_node_ids.
  if (getNodeId() < 0)
    return emitOpError("node_id must be non-negative");

  return success();
}

//===----------------------------------------------------------------------===//
// GraphOp::verify
//===----------------------------------------------------------------------===//
//
// Verifies structural invariants for the top-level graph container:
//   - num_nodes >= 0 (non-negative; 0 is valid for empty graphs)
//   - num_edges >= 0
//   - dispatch_mode is "performance", "cost", or "balanced"
//   - iteration_count (if present) is non-negative
//   - DAG acyclicity: no cycles among kernel nodes connected by edges
//
// The DAG acyclicity check is the most complex verification. It uses Kahn's
// algorithm (topological sort via in-degree tracking) to detect cycles:
//   1. Collect all kernel_launch and fused_kernel node IDs from the body
//   2. Collect all data_dep and anti_dep edges (source → dest node IDs)
//   3. Build an adjacency list and in-degree map
//   4. Process nodes with zero in-degree, decrementing neighbor in-degrees
//   5. If not all nodes are processed, a cycle exists
//
LogicalResult GraphOp::verify() {
  // num_nodes must be non-negative. A value of 0 is valid and represents an
  // empty graph (e.g., during initial construction before kernels are added,
  // or a degenerate trace capture with no kernel launches).
  if (getNumNodes() < 0)
    return emitOpError("num_nodes must be non-negative");

  // num_edges must be non-negative. A graph with zero edges is valid (all
  // kernels are independent and can be scheduled concurrently).
  if (getNumEdges() < 0)
    return emitOpError("num_edges must be non-negative");

  // dispatch_mode must be one of the three supported optimization objectives:
  //   "performance" — Minimize end-to-end latency
  //   "cost"        — Minimize total GPU-seconds (device utilization)
  //   "balanced"    — Weighted combination of performance and cost
  auto mode = getDispatchMode();
  if (mode != "performance" && mode != "cost" && mode != "balanced")
    return emitOpError(
        "dispatch_mode must be 'performance', 'cost', or 'balanced'");

  // If iteration_count is present, it must be non-negative. The feedback
  // controller enforces the maximum cap (TRITON_FEEDBACK_MAX_ITERS, default
  // 20) at runtime; the verifier ensures basic validity.
  if (auto iterCount = getIterationCount()) {
    if (*iterCount < 0)
      return emitOpError("iteration_count must be non-negative");
  }

  // -----------------------------------------------------------------------
  // DAG Acyclicity Check via Kahn's Algorithm
  // -----------------------------------------------------------------------
  // The kernel graph must be a directed acyclic graph (DAG). Cycles would
  // make topological ordering impossible and break the scheduler's critical-
  // path analysis, multi-stream emission, and code generation bridge.
  //
  // We collect kernel node IDs (from kernel_launch and fused_kernel ops)
  // and edges (from data_dep and anti_dep ops), then run Kahn's algorithm
  // to detect cycles. Transfer ops are excluded because they represent
  // physical data movement inserted by the dispatch layer and do not
  // introduce logical dependency cycles.
  // -----------------------------------------------------------------------

  // Phase 1: Collect kernel node IDs and edges from the graph body.
  // Using SmallSet for efficient membership testing during edge validation.
  llvm::SmallSet<int32_t, 16> nodeIds;
  llvm::SmallVector<std::pair<int32_t, int32_t>, 16> edges;

  for (Operation &op : getBody().front()) {
    if (auto launch = dyn_cast<KernelLaunchOp>(&op)) {
      nodeIds.insert(launch.getNodeId());
    } else if (auto fused = dyn_cast<FusedKernelOp>(&op)) {
      nodeIds.insert(fused.getNodeId());
    } else if (auto dep = dyn_cast<DataDepOp>(&op)) {
      edges.push_back({dep.getSourceNodeId(), dep.getDestNodeId()});
    } else if (auto anti = dyn_cast<AntiDepOp>(&op)) {
      edges.push_back({anti.getSourceNodeId(), anti.getDestNodeId()});
    }
  }

  // Skip cycle detection if the graph body has no kernel nodes.
  // This is a valid state during graph construction before kernels are added.
  if (nodeIds.empty())
    return success();

  // Phase 2: Build adjacency list and in-degree map.
  // We use DenseMap for O(1) lookup and SmallVector for compact adjacency
  // lists, matching the expected graph sizes (typically < 100 nodes).
  llvm::DenseMap<int32_t, llvm::SmallVector<int32_t, 4>> adjList;
  llvm::DenseMap<int32_t, unsigned> inDegree;

  // Initialize every known node with zero in-degree and an empty adjacency
  // list. This ensures all nodes participate in Kahn's algorithm even if
  // they have no incoming or outgoing edges.
  for (int32_t id : nodeIds) {
    adjList[id];       // Default-construct empty SmallVector
    inDegree[id] = 0;  // Initialize in-degree to zero
  }

  // Process edges, only adding edges that connect known kernel nodes.
  // Edges referencing node IDs not present in the graph (e.g., stale
  // references after fusion) are silently skipped — they don't affect
  // the cycle detection of the current graph topology.
  for (auto &[src, dst] : edges) {
    if (nodeIds.count(src) && nodeIds.count(dst)) {
      adjList[src].push_back(dst);
      inDegree[dst]++;
    }
  }

  // Phase 3: Kahn's algorithm — process nodes with zero in-degree.
  // We use a SmallVector as a LIFO worklist (stack). Using LIFO vs FIFO
  // does not affect cycle detection correctness — it only changes the
  // topological ordering. Both correctly identify whether all nodes can
  // be processed (DAG) or not (cycle exists).
  llvm::SmallVector<int32_t, 16> worklist;
  for (auto &[id, deg] : inDegree) {
    if (deg == 0)
      worklist.push_back(id);
  }

  unsigned processedCount = 0;
  while (!worklist.empty()) {
    int32_t node = worklist.pop_back_val();
    processedCount++;

    // For each outgoing neighbor, decrement its in-degree. If the
    // neighbor's in-degree reaches zero, it has no remaining unprocessed
    // predecessors and can be added to the worklist.
    for (int32_t neighbor : adjList[node]) {
      if (--inDegree[neighbor] == 0)
        worklist.push_back(neighbor);
    }
  }

  // If we could not process all nodes, there must be a cycle — at least
  // one node has a non-zero in-degree due to a back-edge in the graph.
  if (processedCount != nodeIds.size())
    return emitOpError("kernel graph contains a cycle");

  return success();
}
