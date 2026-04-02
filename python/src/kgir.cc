//===-- kgir.cc - PyBind11 bindings for TritonKGIR dialect -------*- C++ -*-===//
//
// Part of the Triton project.
//
// Provides PyBind11 bindings exposing KGIR (Kernel Graph IR) construction,
// operation manipulation, type construction, attribute manipulation, graph
// traversal, annotation read/write, and module serialization to Python.
//
// Defines init_triton_kgir(py::module &&m) which is called from main.cc to
// register the 'kgir' submodule for Python-level KGIR manipulation.
//
// Follows the coding patterns established by gluon_ir.cc (GluonOpBuilder)
// and ir.cc (TritonOpBuilder base class).
//
//===----------------------------------------------------------------------===//

#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/KGIRToTTIR/Passes.h"
#include "triton/Dialect/TritonKGIR/IR/Dialect.h"
#include "triton/Dialect/TritonKGIR/Transforms/Passes.h"

using namespace mlir;
namespace py = pybind11;
namespace kgir = mlir::triton::kgir;

//===----------------------------------------------------------------------===//
// KGIROpBuilder — Builder class for constructing KGIR MLIR operations
//===----------------------------------------------------------------------===//
//
// Inherits from TritonOpBuilder (defined in ir.h) following the same pattern
// as GluonOpBuilder in gluon_ir.cc. Provides type construction, attribute
// construction, and operation creation methods for the TritonKGIR dialect.
//

struct KGIROpBuilder : public TritonOpBuilder {
  using TritonOpBuilder::TritonOpBuilder;

  // ---- Type construction methods ----
  // These construct KGIR MLIR types for hardware profiles, node metadata,
  // and performance annotations. Parameters match the TableGen definitions
  // in TritonKGIRTypes.td.

  /// Construct a HardwareProfileType describing a GPU device's capabilities.
  /// 12 parameters matching the HardwareProfile concrete schema from the AAP.
  /// Returns mlir::Type (base class) for pybind11 compatibility.
  Type getHardwareProfileType(
      const std::string &vendor, const std::string &archGeneration,
      int32_t smCount, int32_t smemPerSmBytes, int32_t registersPerSm,
      int64_t globalMemoryBytes, double memoryBandwidthGbps,
      double computeThroughputTflops, int32_t warpSize,
      int32_t maxConcurrentStreams, const std::string &interconnectType,
      double interconnectBandwidthGbps) {
    return kgir::HardwareProfileType::get(
        getContext(), StringRef(vendor), StringRef(archGeneration), smCount,
        smemPerSmBytes, registersPerSm, globalMemoryBytes,
        memoryBandwidthGbps, computeThroughputTflops, warpSize,
        maxConcurrentStreams, StringRef(interconnectType),
        interconnectBandwidthGbps);
  }

  /// Construct a NodeMetadataType capturing per-kernel resource usage.
  /// 8 parameters: tensor arg count, grid dims (x,y,z), shared memory,
  /// register pressure, read/write counts.
  /// Returns mlir::Type (base class) for pybind11 compatibility.
  Type getNodeMetadataType(
      int32_t numTensorArgs, int32_t gridDimX, int32_t gridDimY,
      int32_t gridDimZ, int32_t sharedMemoryBytes, int32_t registerPressure,
      int32_t numReads, int32_t numWrites) {
    return kgir::NodeMetadataType::get(getContext(), numTensorArgs, gridDimX,
                                       gridDimY, gridDimZ, sharedMemoryBytes,
                                       registerPressure, numReads, numWrites);
  }

  /// Construct a PerformanceAnnotationType for measured runtime metrics.
  /// 6 double parameters: wall clock, memory throughput, occupancy,
  /// bandwidth utilization, launch overhead, transfer time.
  /// Returns mlir::Type (base class) for pybind11 compatibility.
  Type getPerformanceAnnotationType(
      double wallClockUs, double memoryThroughputGbps, double occupancy,
      double bandwidthUtilization, double launchOverheadUs,
      double transferTimeUs) {
    return kgir::PerformanceAnnotationType::get(
        getContext(), wallClockUs, memoryThroughputGbps, occupancy,
        bandwidthUtilization, launchOverheadUs, transferTimeUs);
  }

  // ---- Attribute construction methods ----
  // These construct KGIR attribute instances for annotating operations.
  // Parameter types match the TableGen definitions in TritonKGIRAttrDefs.td.

  /// Construct a MemoryAccessPatternAttr describing a tensor's access pattern.
  /// pattern_type: "read", "write", or "readwrite"
  /// Returns mlir::Attribute (base class) for pybind11 compatibility.
  Attribute createMemoryAccessPatternAttr(
      const std::string &patternType, int64_t tensorIndex, bool isContiguous,
      int64_t accessSizeBytes) {
    return kgir::MemoryAccessPatternAttr::get(getContext(),
                                              StringRef(patternType),
                                              tensorIndex, isContiguous,
                                              accessSizeBytes);
  }

  /// Construct a HardwareTargetAnnotationAttr for dispatch assignment.
  /// Returns mlir::Attribute (base class) for pybind11 compatibility.
  Attribute createHardwareTargetAnnotationAttr(
      int64_t deviceId, const std::string &vendor, const std::string &arch,
      int64_t streamId) {
    return kgir::HardwareTargetAnnotationAttr::get(
        getContext(), deviceId, StringRef(vendor), StringRef(arch), streamId);
  }

  /// Construct a RuntimePerformanceAttr with measured performance data.
  /// Returns mlir::Attribute (base class) for pybind11 compatibility.
  Attribute createRuntimePerformanceAttr(
      double wallClockUs, double memoryThroughputGbps, double occupancy,
      double launchOverheadUs, int64_t targetDeviceId, int64_t iteration) {
    return kgir::RuntimePerformanceAttr::get(
        getContext(), wallClockUs, memoryThroughputGbps, occupancy,
        launchOverheadUs, targetDeviceId, iteration);
  }

  /// Construct a FusionDecisionAttr recording a fusion accept/reject decision.
  /// fusion_type: "producer_consumer" or "sibling"
  /// Returns mlir::Attribute (base class) for pybind11 compatibility.
  Attribute createFusionDecisionAttr(
      bool isFused, const std::string &fusionType, const std::string &reason,
      double estimatedSpeedup, int64_t targetDeviceId) {
    return kgir::FusionDecisionAttr::get(getContext(), isFused,
                                         StringRef(fusionType),
                                         StringRef(reason), estimatedSpeedup,
                                         targetDeviceId);
  }

  // ---- Operation creation methods ----
  // These create KGIR operations in the current insertion point.
  // The create<OpTy>() template from TritonOpBuilder is used, which
  // automatically injects the current location via getLastLoc().
  // Parameters match the actual TableGen op definitions in TritonKGIROps.td.

  /// Create a ttkgir.kernel_launch operation representing a single kernel
  /// in the graph. Optional attributes (memory_access_patterns, hardware_target,
  /// runtime_perf, tensor_shapes) are left unset and populated later by the
  /// fusion analysis, dispatch, and profiler passes respectively.
  void createKernelLaunch(const std::string &kernelName,
                          const std::vector<int64_t> &gridDims,
                          int32_t numWarps, int32_t sharedMemoryBytes,
                          int32_t registerPressure, int32_t nodeId) {
    auto &b = getBuilder();
    create<kgir::KernelLaunchOp>(
        b.getStringAttr(kernelName), b.getDenseI64ArrayAttr(gridDims),
        b.getI32IntegerAttr(numWarps), b.getI32IntegerAttr(sharedMemoryBytes),
        b.getI32IntegerAttr(registerPressure),
        /*memory_access_patterns=*/kgir::MemoryAccessPatternAttr{},
        /*hardware_target=*/kgir::HardwareTargetAnnotationAttr{},
        /*runtime_perf=*/kgir::RuntimePerformanceAttr{},
        /*tensor_shapes=*/DenseI64ArrayAttr{},
        b.getI32IntegerAttr(nodeId));
  }

  /// Create a ttkgir.data_dep operation representing a data dependency edge.
  /// source_node_id produces data consumed by dest_node_id through
  /// tensor_index with dependency type dep_type (e.g., "producer_consumer").
  void createDataDep(int32_t sourceNodeId, int32_t destNodeId,
                     int32_t tensorIndex, const std::string &depType) {
    auto &b = getBuilder();
    create<kgir::DataDepOp>(b.getI32IntegerAttr(sourceNodeId),
                            b.getI32IntegerAttr(destNodeId),
                            b.getI32IntegerAttr(tensorIndex),
                            b.getStringAttr(depType));
  }

  /// Create a ttkgir.anti_dep operation representing an anti-dependency edge.
  /// conflict_type describes the conflict (e.g., "write_after_read").
  void createAntiDep(int32_t sourceNodeId, int32_t destNodeId,
                     int32_t tensorIndex, const std::string &conflictType) {
    auto &b = getBuilder();
    create<kgir::AntiDepOp>(b.getI32IntegerAttr(sourceNodeId),
                            b.getI32IntegerAttr(destNodeId),
                            b.getI32IntegerAttr(tensorIndex),
                            b.getStringAttr(conflictType));
  }

  /// Create a ttkgir.transfer operation representing a cross-device data
  /// transfer. The optional estimated_latency_us is left unset initially.
  void createTransfer(int32_t sourceDeviceId, int32_t destDeviceId,
                      int32_t tensorIndex, int64_t transferSizeBytes,
                      const std::string &interconnectType,
                      int32_t sourceNodeId, int32_t destNodeId) {
    auto &b = getBuilder();
    create<kgir::TransferOp>(
        b.getI32IntegerAttr(sourceDeviceId),
        b.getI32IntegerAttr(destDeviceId), b.getI32IntegerAttr(tensorIndex),
        b.getI64IntegerAttr(transferSizeBytes),
        b.getStringAttr(interconnectType),
        /*estimated_latency_us=*/FloatAttr{},
        b.getI32IntegerAttr(sourceNodeId), b.getI32IntegerAttr(destNodeId));
  }

  /// Create a ttkgir.fused_kernel operation representing the result of
  /// fusing multiple kernel nodes. Optional attributes (hardware_target,
  /// runtime_perf, fusion_decision) are populated by later passes.
  void createFusedKernel(const std::string &fusedName,
                         const std::vector<int32_t> &fusedNodeIds,
                         const std::string &fusionType,
                         const std::vector<int64_t> &combinedGridDims,
                         int32_t combinedSharedMemoryBytes,
                         int32_t combinedRegisterPressure,
                         int32_t targetDeviceId, int32_t nodeId) {
    auto &b = getBuilder();
    create<kgir::FusedKernelOp>(
        b.getStringAttr(fusedName),
        DenseI32ArrayAttr::get(getContext(), fusedNodeIds),
        b.getStringAttr(fusionType), b.getDenseI64ArrayAttr(combinedGridDims),
        b.getI32IntegerAttr(combinedSharedMemoryBytes),
        b.getI32IntegerAttr(combinedRegisterPressure),
        b.getI32IntegerAttr(targetDeviceId),
        /*hardware_target=*/kgir::HardwareTargetAnnotationAttr{},
        /*runtime_perf=*/kgir::RuntimePerformanceAttr{},
        /*fusion_decision=*/kgir::FusionDecisionAttr{},
        b.getI32IntegerAttr(nodeId));
  }

  /// Create a ttkgir.graph operation — the top-level container with a
  /// SingleBlock, NoTerminator region for holding kernel, dependency,
  /// transfer, and fused kernel operations. The num_devices parameter is
  /// stored as a custom attribute for the dispatch layer; num_nodes and
  /// num_edges are initialized to zero and updated as operations are added.
  Operation *createGraph(const std::string &graphName, int32_t numDevices,
                         const std::string &dispatchMode) {
    auto &b = getBuilder();
    auto op = create<kgir::GraphOp>(
        b.getStringAttr(graphName),
        b.getI32IntegerAttr(0),  // num_nodes — populated as kernels are added
        b.getI32IntegerAttr(0),  // num_edges — populated as deps are added
        b.getStringAttr(""),     // hardware_profiles — populated by dispatch
        b.getStringAttr(dispatchMode),
        /*is_converged=*/BoolAttr{},
        /*iteration_count=*/IntegerAttr{});
    // Store num_devices as a custom attribute for dispatch layer consumption
    op->setAttr("num_devices", b.getI32IntegerAttr(numDevices));
    // Ensure the region has a block (SingleBlock trait requirement)
    Region &body = op.getBody();
    if (body.empty())
      body.emplaceBlock();
    return op.getOperation();
  }
};

//===----------------------------------------------------------------------===//
// Graph Traversal Utility Functions
//===----------------------------------------------------------------------===//
//
// These standalone functions operate on KGIR graph operations to provide
// structural queries: operation enumeration, topological sorting, and
// predecessor/successor navigation. They walk the operations in the graph's
// body region to extract the requested information.
//

/// Extract a 32-bit integer attribute value from an operation by name.
/// Returns 0 if the attribute is missing or not an IntegerAttr.
static int32_t getI32Attr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return static_cast<int32_t>(attr.getInt());
  return 0;
}

/// Get all ttkgir.kernel_launch operations from a graph operation's body.
static std::vector<Operation *> getKernelLaunchOps(Operation *graphOp) {
  std::vector<Operation *> result;
  if (!graphOp || graphOp->getNumRegions() == 0)
    return result;
  Region &body = graphOp->getRegion(0);
  if (body.empty())
    return result;
  for (Operation &op : body.front()) {
    if (isa<kgir::KernelLaunchOp>(op))
      result.push_back(&op);
  }
  return result;
}

/// Get all ttkgir.data_dep operations from a graph operation's body.
static std::vector<Operation *> getDataDepOps(Operation *graphOp) {
  std::vector<Operation *> result;
  if (!graphOp || graphOp->getNumRegions() == 0)
    return result;
  Region &body = graphOp->getRegion(0);
  if (body.empty())
    return result;
  for (Operation &op : body.front()) {
    if (isa<kgir::DataDepOp>(op))
      result.push_back(&op);
  }
  return result;
}

/// Get all ttkgir.fused_kernel operations from a graph operation's body.
static std::vector<Operation *> getFusedKernelOps(Operation *graphOp) {
  std::vector<Operation *> result;
  if (!graphOp || graphOp->getNumRegions() == 0)
    return result;
  Region &body = graphOp->getRegion(0);
  if (body.empty())
    return result;
  for (Operation &op : body.front()) {
    if (isa<kgir::FusedKernelOp>(op))
      result.push_back(&op);
  }
  return result;
}

/// Perform topological sort of kernel nodes based on data dependencies.
/// Uses Kahn's algorithm over the DAG edges. Returns node IDs in
/// topological order. If the graph contains cycles (invalid), nodes
/// involved in cycles are omitted from the result.
static std::vector<int32_t> topologicalSortNodes(Operation *graphOp) {
  std::vector<int32_t> result;
  if (!graphOp || graphOp->getNumRegions() == 0)
    return result;
  Region &body = graphOp->getRegion(0);
  if (body.empty())
    return result;

  // Collect all unique node IDs from kernel_launch and fused_kernel ops
  std::unordered_set<int32_t> allNodes;
  for (Operation &op : body.front()) {
    if (isa<kgir::KernelLaunchOp>(op) || isa<kgir::FusedKernelOp>(op))
      allNodes.insert(getI32Attr(&op, "node_id"));
  }

  // Build adjacency list and in-degree map from data_dep edges
  std::unordered_map<int32_t, std::vector<int32_t>> adj;
  std::unordered_map<int32_t, int32_t> inDegree;
  for (int32_t node : allNodes) {
    adj[node] = {};
    inDegree[node] = 0;
  }
  for (Operation &op : body.front()) {
    if (isa<kgir::DataDepOp>(op)) {
      int32_t src = getI32Attr(&op, "source_node_id");
      int32_t dst = getI32Attr(&op, "dest_node_id");
      adj[src].push_back(dst);
      inDegree[dst]++;
    }
  }

  // Kahn's algorithm: process nodes with zero in-degree first
  std::queue<int32_t> readyQueue;
  for (int32_t node : allNodes) {
    if (inDegree[node] == 0)
      readyQueue.push(node);
  }
  while (!readyQueue.empty()) {
    int32_t curr = readyQueue.front();
    readyQueue.pop();
    result.push_back(curr);
    for (int32_t neighbor : adj[curr]) {
      inDegree[neighbor]--;
      if (inDegree[neighbor] == 0)
        readyQueue.push(neighbor);
    }
  }
  return result;
}

/// Get predecessor node IDs for a given node based on data dependencies.
/// A predecessor of nodeId is any node that is a source in a data_dep
/// edge whose destination is nodeId.
static std::vector<int32_t> getPredecessors(Operation *graphOp,
                                            int32_t nodeId) {
  std::vector<int32_t> result;
  if (!graphOp || graphOp->getNumRegions() == 0)
    return result;
  Region &body = graphOp->getRegion(0);
  if (body.empty())
    return result;
  for (Operation &op : body.front()) {
    if (isa<kgir::DataDepOp>(op)) {
      if (getI32Attr(&op, "dest_node_id") == nodeId)
        result.push_back(getI32Attr(&op, "source_node_id"));
    }
  }
  return result;
}

/// Get successor node IDs for a given node based on data dependencies.
/// A successor of nodeId is any node that is a destination in a data_dep
/// edge whose source is nodeId.
static std::vector<int32_t> getSuccessors(Operation *graphOp,
                                          int32_t nodeId) {
  std::vector<int32_t> result;
  if (!graphOp || graphOp->getNumRegions() == 0)
    return result;
  Region &body = graphOp->getRegion(0);
  if (body.empty())
    return result;
  for (Operation &op : body.front()) {
    if (isa<kgir::DataDepOp>(op)) {
      if (getI32Attr(&op, "source_node_id") == nodeId)
        result.push_back(getI32Attr(&op, "dest_node_id"));
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Annotation Read/Write Utility Functions
//===----------------------------------------------------------------------===//
//
// These functions enable the closed-loop feedback mechanism by writing
// measured performance data back to KGIR node annotations and reading
// them for cost model calibration and convergence detection.
//

/// Set runtime_perf attribute on a kernel_launch or fused_kernel operation.
/// Called by the runtime profiler after each execution iteration to record
/// measured performance for the feedback controller.
static void setRuntimePerformance(Operation *op, double wallClockUs,
                                  double memThroughput, double occupancy,
                                  double launchOverhead,
                                  int64_t targetDeviceId, int64_t iteration) {
  auto *ctx = op->getContext();
  auto attr = kgir::RuntimePerformanceAttr::get(
      ctx, wallClockUs, memThroughput, occupancy, launchOverhead,
      targetDeviceId, iteration);
  op->setAttr("runtime_perf", attr);
}

/// Get runtime_perf attribute from a kernel_launch or fused_kernel operation.
/// Returns a Python dict with field names matching the AAP spec if the
/// attribute is set, or Python None if no runtime performance data exists.
static py::object getRuntimePerformance(Operation *op) {
  auto attr =
      op->getAttrOfType<kgir::RuntimePerformanceAttr>("runtime_perf");
  if (!attr)
    return py::none();
  py::dict result;
  result["wall_clock_us"] = attr.getWallClockUs();
  result["memory_throughput_gbps"] = attr.getMemoryThroughputGbps();
  result["occupancy"] = attr.getOccupancy();
  result["launch_overhead_us"] = attr.getLaunchOverheadUs();
  result["target_device_id"] = attr.getTargetDeviceId();
  result["iteration"] = attr.getIteration();
  return result;
}

/// Set hardware_target attribute on a kernel_launch or fused_kernel operation.
/// Written by the dispatch layer to assign a specific device and stream.
static void setHardwareTarget(Operation *op, int64_t deviceId,
                              const std::string &vendor,
                              const std::string &arch, int64_t streamId) {
  auto *ctx = op->getContext();
  auto attr = kgir::HardwareTargetAnnotationAttr::get(
      ctx, deviceId, StringRef(vendor), StringRef(arch), streamId);
  op->setAttr("hardware_target", attr);
}

/// Set fusion_decision attribute on a fused_kernel operation.
/// Recorded by the fusion analysis engine for closed-loop reversal logic.
static void setFusionDecision(Operation *op, bool isFused,
                              const std::string &fusionType,
                              const std::string &reason,
                              double estimatedSpeedup,
                              int64_t targetDeviceId) {
  auto *ctx = op->getContext();
  auto attr = kgir::FusionDecisionAttr::get(
      ctx, isFused, StringRef(fusionType), StringRef(reason),
      estimatedSpeedup, targetDeviceId);
  op->setAttr("fusion_decision", attr);
}

//===----------------------------------------------------------------------===//
// KGIR Module Serialization
//===----------------------------------------------------------------------===//

/// Serialize a KGIR module (or any MLIR Operation) to its textual
/// representation. Used by the code generation bridge for debugging
/// and by the graph cache for signature computation.
static std::string serializeKGIRModule(Operation *moduleOp) {
  std::string str;
  llvm::raw_string_ostream os(str);
  moduleOp->print(os);
  return str;
}

//===----------------------------------------------------------------------===//
// Body Access Utilities
//===----------------------------------------------------------------------===//
//
// The existing pybind11 bindings for Region do not expose a get_block(idx)
// method.  These helpers allow Python tests and graph construction code to
// obtain the body Block* of a ModuleOp or a GraphOp so that the builder's
// insertion point can be placed inside them.
//

/// Return the body Block of a ModuleOp.  ModuleOp always has a single region
/// with a single block.
static Block *getModuleBody(ModuleOp moduleOp) {
  return moduleOp.getBody();
}

/// Return the body Block of a GraphOp.  GraphOp has SingleBlock trait,
/// so getBody() returns the front block of region(0).
static Block *getGraphBody(Operation *graphOp) {
  if (!graphOp || graphOp->getNumRegions() == 0)
    throw std::runtime_error("getGraphBody: operation has no regions");
  Region &region = graphOp->getRegion(0);
  if (region.empty())
    throw std::runtime_error("getGraphBody: region has no blocks");
  return &region.front();
}

//===----------------------------------------------------------------------===//
// init_triton_kgir — PyBind11 Module Registration
//===----------------------------------------------------------------------===//
//
// Entry point called from main.cc via:
//   init_triton_kgir(m.def_submodule("kgir"))
//
// Registers the KGIROpBuilder class, graph traversal functions, annotation
// read/write functions, and serialization utilities in the Python 'kgir'
// submodule of triton._C.libtriton.
//

void init_triton_kgir(py::module &&m) {
  using ret = py::return_value_policy;

  // ---- Dialect loading utility ----
  // Allows explicit KGIR dialect loading into an existing MLIRContext.
  // Typically not needed if ir.load_dialects() already includes KGIR,
  // but provided for standalone / testing scenarios.
  m.def("load_dialect",
        [](MLIRContext &context) {
          context.getOrLoadDialect<kgir::TritonKGIRDialect>();
        },
        py::arg("context"),
        "Load the TritonKGIR dialect into the given MLIRContext.");

  // ---- KGIROpBuilder class registration ----
  py::class_<KGIROpBuilder, TritonOpBuilder>(m, "KGIROpBuilder",
                                             py::module_local(),
                                             py::dynamic_attr())
      .def(py::init<MLIRContext *>())

      // Type construction methods
      .def("get_hardware_profile_type",
           &KGIROpBuilder::getHardwareProfileType, py::arg("vendor"),
           py::arg("arch_generation"), py::arg("sm_count"),
           py::arg("smem_per_sm_bytes"), py::arg("registers_per_sm"),
           py::arg("global_memory_bytes"), py::arg("memory_bandwidth_gbps"),
           py::arg("compute_throughput_tflops"), py::arg("warp_size"),
           py::arg("max_concurrent_streams"), py::arg("interconnect_type"),
           py::arg("interconnect_bandwidth_gbps"))
      .def("get_node_metadata_type", &KGIROpBuilder::getNodeMetadataType,
           py::arg("num_tensor_args"), py::arg("grid_dim_x"),
           py::arg("grid_dim_y"), py::arg("grid_dim_z"),
           py::arg("shared_memory_bytes"), py::arg("register_pressure"),
           py::arg("num_reads"), py::arg("num_writes"))
      .def("get_performance_annotation_type",
           &KGIROpBuilder::getPerformanceAnnotationType,
           py::arg("wall_clock_us"), py::arg("memory_throughput_gbps"),
           py::arg("occupancy"), py::arg("bandwidth_utilization"),
           py::arg("launch_overhead_us"), py::arg("transfer_time_us"))

      // Attribute construction methods
      .def("create_memory_access_pattern_attr",
           &KGIROpBuilder::createMemoryAccessPatternAttr,
           py::arg("pattern_type"), py::arg("tensor_index"),
           py::arg("is_contiguous"), py::arg("access_size_bytes"))
      .def("create_hardware_target_annotation_attr",
           &KGIROpBuilder::createHardwareTargetAnnotationAttr,
           py::arg("device_id"), py::arg("vendor"), py::arg("arch"),
           py::arg("stream_id"))
      .def("create_runtime_performance_attr",
           &KGIROpBuilder::createRuntimePerformanceAttr,
           py::arg("wall_clock_us"), py::arg("memory_throughput_gbps"),
           py::arg("occupancy"), py::arg("launch_overhead_us"),
           py::arg("target_device_id"), py::arg("iteration"))
      .def("create_fusion_decision_attr",
           &KGIROpBuilder::createFusionDecisionAttr, py::arg("is_fused"),
           py::arg("fusion_type"), py::arg("reason"),
           py::arg("estimated_speedup"), py::arg("target_device_id"))

      // Operation creation methods
      .def("create_kernel_launch", &KGIROpBuilder::createKernelLaunch,
           py::arg("kernel_name"), py::arg("grid_dims"),
           py::arg("num_warps"), py::arg("shared_memory_bytes"),
           py::arg("register_pressure"), py::arg("node_id"))
      .def("create_data_dep", &KGIROpBuilder::createDataDep,
           py::arg("source_node_id"), py::arg("dest_node_id"),
           py::arg("tensor_index"), py::arg("dep_type"))
      .def("create_anti_dep", &KGIROpBuilder::createAntiDep,
           py::arg("source_node_id"), py::arg("dest_node_id"),
           py::arg("tensor_index"), py::arg("conflict_type"))
      .def("create_transfer", &KGIROpBuilder::createTransfer,
           py::arg("source_device_id"), py::arg("dest_device_id"),
           py::arg("tensor_index"), py::arg("transfer_size_bytes"),
           py::arg("interconnect_type"), py::arg("source_node_id"),
           py::arg("dest_node_id"))
      .def("create_fused_kernel", &KGIROpBuilder::createFusedKernel,
           py::arg("fused_name"), py::arg("fused_node_ids"),
           py::arg("fusion_type"), py::arg("combined_grid_dims"),
           py::arg("combined_shared_memory_bytes"),
           py::arg("combined_register_pressure"),
           py::arg("target_device_id"), py::arg("node_id"))
      .def("create_graph", &KGIROpBuilder::createGraph,
           py::arg("graph_name"), py::arg("num_devices"),
           py::arg("dispatch_mode"), ret::reference);

  // ---- Graph traversal utility functions ----
  m.def("get_kernel_launch_ops", &getKernelLaunchOps, py::arg("graph_op"));
  m.def("get_data_dep_ops", &getDataDepOps, py::arg("graph_op"));
  m.def("get_fused_kernel_ops", &getFusedKernelOps, py::arg("graph_op"));
  m.def("topological_sort_nodes", &topologicalSortNodes,
        py::arg("graph_op"));
  m.def("get_predecessors", &getPredecessors, py::arg("graph_op"),
        py::arg("node_id"));
  m.def("get_successors", &getSuccessors, py::arg("graph_op"),
        py::arg("node_id"));

  // ---- Annotation read/write functions ----
  m.def("set_runtime_performance", &setRuntimePerformance, py::arg("op"),
        py::arg("wall_clock_us"), py::arg("memory_throughput_gbps"),
        py::arg("occupancy"), py::arg("launch_overhead_us"),
        py::arg("target_device_id"), py::arg("iteration"));
  m.def("get_runtime_performance", &getRuntimePerformance, py::arg("op"));
  m.def("set_hardware_target", &setHardwareTarget, py::arg("op"),
        py::arg("device_id"), py::arg("vendor"), py::arg("arch"),
        py::arg("stream_id"));
  m.def("set_fusion_decision", &setFusionDecision, py::arg("op"),
        py::arg("is_fused"), py::arg("fusion_type"), py::arg("reason"),
        py::arg("estimated_speedup"), py::arg("target_device_id"));

  // ---- Body access utilities ----
  m.def(
      "get_module_body",
      [](ModuleOp module) -> Block * { return getModuleBody(module); },
      py::arg("module"), py::return_value_policy::reference,
      "Return the body Block of a ModuleOp.");
  m.def("get_graph_body", &getGraphBody, py::arg("graph_op"),
        py::return_value_policy::reference,
        "Return the body Block of a GraphOp.");

  // ---- KGIR module serialization ----
  m.def("serialize_kgir_module", &serializeKGIRModule,
        py::arg("module_op"));
}
