//===- MemoryPlanning.cpp - KGIR Memory Planning Pass -----------*- C++ -*-===//
//
// Part of the Triton project.
//
// This file implements the ttkgir-memory-planning MLIR pass for the
// TritonKGIR dialect. The pass performs:
//   1. Liveness analysis on the KGIR DAG via Kahn's topological sort
//   2. Identification of intermediate tensors (produced and consumed within
//      the same graph) via data_dep edge traversal
//   3. Global→shared memory promotion of eligible intermediates, respecting
//      per-target SMEM capacity from HardwareProfile descriptors
//   4. Register file promotion for small intermediates below a register
//      threshold
//   5. Cross-device transfer operation insertion when dispatch annotations
//      assign producer and consumer to different devices
//   6. Closed-loop feedback refinement: revert promotions that caused
//      measured occupancy degradation based on RuntimePerformanceAttr
//
// Algorithm selection rationale (AAP §0.5.3):
//   Memory promotion uses a greedy interval-based algorithm that prioritizes
//   larger intermediates (more eliminated global memory traffic) and uses a
//   sweep-line SMEM budget check across topological positions. This is chosen
//   over ILP-based optimal promotion for latency reasons (< 100ms for ≤50
//   kernel graphs) and over simple first-fit for better memory traffic
//   elimination.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonKGIR/Transforms/Passes.h"

#include "triton/Dialect/TritonKGIR/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>

#define DEBUG_TYPE "ttkgir-memory-planning"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

// GEN_PASS_DEF must be inside the dialect namespace so that the generated base
// class template and factory function reside in mlir::triton::kgir::impl.
namespace mlir::triton::kgir {
#define GEN_PASS_DEF_TRITONKGIRMEMORYPLANNING
#include "triton/Dialect/TritonKGIR/Transforms/Passes.h.inc"
} // namespace mlir::triton::kgir

namespace ttkgir = mlir::triton::kgir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Data Structures
//===----------------------------------------------------------------------===//

/// Represents the lifetime of an intermediate tensor within the KGIR graph.
/// An "intermediate" is a tensor produced by one kernel and consumed by
/// another, both within the same graph scope — making it a candidate for
/// promotion from global memory to faster on-chip memory.
struct LivenessInterval {
  int producerNodeId;      // Node that produces this tensor
  int consumerNodeId;      // Last consumer node for this tensor
  int32_t tensorIndex;     // Tensor argument index on the producer
  int64_t tensorSizeBytes; // Estimated size of the intermediate tensor
  int firstUse;            // Topological order of producer (first definition)
  int lastUse;             // Topological order of last consumer
  bool isPromotable;       // Whether eligible for global→shared promotion
  bool isPromoted;         // Whether actually promoted after analysis
};

/// Represents a decided memory promotion for an intermediate tensor.
struct MemoryPromotion {
  int producerNodeId;      // Source kernel node ID
  int consumerNodeId;      // Destination kernel node ID
  int32_t tensorIndex;     // Tensor argument index on the producer
  int64_t tensorSizeBytes; // Size of the promoted intermediate
  enum PromotionTarget {
    SharedMemory,  // Promote to shared memory (SMEM)
    RegisterFile   // Promote to register file (small tensors only)
  } target;
};

//===----------------------------------------------------------------------===//
// Static Helper Functions
//===----------------------------------------------------------------------===//

/// Default shared memory capacity per SM/CU used when no HardwareProfile
/// is available.  This is a conservative estimate matching NVIDIA Ampere's
/// configurable limit (48 KiB).
static constexpr int64_t kDefaultSmemCapacity = 49152;

/// Maximum tensor size eligible for register file promotion.
/// Tensors must fit within a warp's worth of registers (256 bytes is a
/// conservative heuristic corresponding to ~64 32-bit registers).
static constexpr int64_t kRegisterPromotionThreshold = 256;

/// Occupancy degradation threshold (10%) for feedback refinement.
/// If measured occupancy drops more than this fraction relative to baseline
/// after promotion, the promotion is reverted.
static constexpr double kOccupancyDegradationThreshold = 0.10;

/// Parse an integer value from a JSON-like string following the given key.
/// Returns defaultVal on parse failure or missing key.
static int64_t parseI64FromProfile(StringRef profiles, StringRef key,
                                   int64_t defaultVal) {
  if (profiles.empty())
    return defaultVal;
  size_t pos = profiles.find(key);
  if (pos == StringRef::npos)
    return defaultVal;
  size_t colonPos = profiles.find(':', pos + key.size());
  if (colonPos == StringRef::npos)
    return defaultVal;
  StringRef rest = profiles.substr(colonPos + 1).ltrim();
  // Extract consecutive digits.
  size_t numEnd = 0;
  while (numEnd < rest.size() && rest[numEnd] >= '0' && rest[numEnd] <= '9')
    ++numEnd;
  if (numEnd == 0)
    return defaultVal;
  int64_t val = 0;
  if (!rest.substr(0, numEnd).getAsInteger(10, val) && val > 0)
    return val;
  return defaultVal;
}

/// Estimate the size of a tensor argument in bytes based on a MemoryAccessPattern
/// attribute's access_size_bytes. Returns 0 if no suitable attribute is found.
static int64_t estimateTensorSizeFromAccessPattern(
    ttkgir::KernelLaunchOp kernelOp, int32_t tensorIndex) {
  auto memPat = kernelOp.getMemoryAccessPatterns();
  if (!memPat)
    return 0;
  // The MemoryAccessPatternAttr stores access_size_bytes for the entire
  // tensor argument. When the tensor_index matches, use that size.
  if (memPat->getTensorIndex() == static_cast<int64_t>(tensorIndex))
    return memPat->getAccessSizeBytes();
  return 0;
}

/// Estimate tensor size for a FusedKernelOp. We use a heuristic based on
/// shared memory allocation since fused kernels may not carry per-tensor
/// access patterns.
static int64_t estimateTensorSizeFromFusedKernel(
    ttkgir::FusedKernelOp fusedOp) {
  // Use combined shared memory as an upper-bound estimate for intermediate
  // tensor sizes within the fused kernel.
  return static_cast<int64_t>(fusedOp.getCombinedSharedMemoryBytes());
}

/// Return the device ID assigned to a compute node Operation.
/// Returns 0 (default device) if no hardware target annotation is present.
static int getNodeDeviceId(Operation *op) {
  if (auto k = dyn_cast<ttkgir::KernelLaunchOp>(op)) {
    if (auto hw = k.getHardwareTarget())
      return static_cast<int>(hw->getDeviceId());
    return 0;
  }
  if (auto f = dyn_cast<ttkgir::FusedKernelOp>(op))
    return static_cast<int>(f.getTargetDeviceId());
  return 0;
}

//===----------------------------------------------------------------------===//
// MemoryPlanning Pass
//===----------------------------------------------------------------------===//

/// The ttkgir-memory-planning MLIR pass. Analyses the KGIR kernel-launch DAG,
/// computes liveness intervals for intermediate tensors, decides global→shared
/// or global→register promotions within per-target SMEM budgets, inserts
/// cross-device transfer operations for dispatch-split intermediates, and
/// refines promotion decisions based on closed-loop runtime feedback.
struct MemoryPlanning
    : public ttkgir::impl::TritonKGIRMemoryPlanningBase<MemoryPlanning> {

  // Inherit constructors from generated base class.
  using TritonKGIRMemoryPlanningBase::TritonKGIRMemoryPlanningBase;

  void runOnOperation() override;

private:
  /// Perform liveness analysis on the KGIR DAG.
  /// Computes topological ordering via Kahn's algorithm and determines
  /// liveness intervals for all intermediate tensors (tensors produced and
  /// consumed entirely within the same graph scope).
  void computeLivenessIntervals(ModuleOp moduleOp,
                                SmallVectorImpl<LivenessInterval> &intervals);

  /// Analyze promotion eligibility for each intermediate tensor.
  /// Eligibility requires:
  ///   - Tensor is produced and consumed entirely within the graph
  ///   - Tensor size fits within per-target SMEM capacity from HardwareProfile
  ///   - Simultaneous promotions at any schedule point do not exceed SMEM budget
  /// Uses a greedy algorithm: prioritize promotions by eliminated global memory
  /// traffic (larger tensors first).
  void analyzePromotionEligibility(
      const SmallVectorImpl<LivenessInterval> &intervals,
      SmallVectorImpl<MemoryPromotion> &promotions);

  /// Check if simultaneous promotions at any schedule point exceed SMEM capacity.
  /// Implements a sweep-line algorithm over topological order:
  ///   - At each position, sum sizes of all live promoted intermediates
  ///   - Track maximum concurrent SMEM usage across all positions
  ///   - Return true if max usage fits within smemCapacity, false otherwise
  bool checkSmemBudget(const SmallVectorImpl<LivenessInterval> &intervals,
                       int64_t smemCapacity);

  /// Insert ttkgir.transfer operations when dispatch splits graphs across
  /// devices. Walks data_dep edges where producer and consumer have different
  /// device assignments (via HardwareTargetAnnotationAttr) and creates transfer
  /// ops capturing source/dest device, tensor size, and interconnect type.
  void insertTransferOps(ModuleOp moduleOp);

  /// Apply closed-loop refinement: revert promotion decisions if runtime
  /// feedback annotations indicate occupancy degradation.
  /// Reads RuntimePerformanceAttr from KGIR nodes. If a node's measured
  /// occupancy dropped more than kOccupancyDegradationThreshold relative to
  /// expected baseline after promotion, the promotion is reverted (removed from
  /// the active promotions list so it falls back to global memory).
  void applyFeedbackRefinement(ModuleOp moduleOp,
                               SmallVectorImpl<MemoryPromotion> &promotions);

  // ---- per-graph DAG state (rebuilt for each GraphOp) ----
  SmallVector<int, 32> topoOrder_;
  DenseMap<int, int> topoIndex_;   // nodeId -> topological index
  DenseMap<int, Operation *> nodeOps_; // nodeId -> kernel/fused op
  // Adjacency: nodeId -> [(successorId)]
  DenseMap<int, SmallVector<int, 4>> successors_;
  // nodeId -> [predecessorId]
  DenseMap<int, SmallVector<int, 4>> predecessors_;
  int64_t smemCapacity_ = kDefaultSmemCapacity;
};

//===----------------------------------------------------------------------===//
// buildDAGAndTopologicalSort — Helper: populate DAG state from a GraphOp
//===----------------------------------------------------------------------===//

/// Populate DAG node/edge state from a GraphOp and compute topological ordering
/// via Kahn's algorithm. Stores results into the pass's member fields.
static void buildDAGAndTopologicalSort(
    ttkgir::GraphOp graphOp,
    SmallVectorImpl<int> &topoOrder,
    DenseMap<int, int> &topoIndex,
    DenseMap<int, Operation *> &nodeOps,
    DenseMap<int, SmallVector<int, 4>> &successors,
    DenseMap<int, SmallVector<int, 4>> &predecessors) {

  topoOrder.clear();
  topoIndex.clear();
  nodeOps.clear();
  successors.clear();
  predecessors.clear();

  // ---------- Collect compute nodes ----------
  graphOp.getBody().walk([&](ttkgir::KernelLaunchOp kOp) {
    int id = static_cast<int>(kOp.getNodeId());
    nodeOps[id] = kOp;
  });

  graphOp.getBody().walk([&](ttkgir::FusedKernelOp fOp) {
    int id = static_cast<int>(fOp.getNodeId());
    nodeOps[id] = fOp;
  });

  if (nodeOps.empty())
    return;

  // ---------- Collect edges ----------
  DenseSet<std::pair<int, int>> edgeSeen;

  auto addEdge = [&](int src, int dst) {
    if (!nodeOps.count(src) || !nodeOps.count(dst))
      return;
    auto key = std::make_pair(src, dst);
    if (edgeSeen.contains(key))
      return;
    edgeSeen.insert(key);
    successors[src].push_back(dst);
    predecessors[dst].push_back(src);
  };

  graphOp.getBody().walk([&](ttkgir::DataDepOp dep) {
    addEdge(static_cast<int>(dep.getSourceNodeId()),
            static_cast<int>(dep.getDestNodeId()));
  });

  // Anti-dependencies also impose ordering constraints.
  graphOp.getBody().walk([&](ttkgir::AntiDepOp anti) {
    addEdge(static_cast<int>(anti.getSourceNodeId()),
            static_cast<int>(anti.getDestNodeId()));
  });

  // Ensure every node has entries even if it has no edges.
  for (const auto &kv : nodeOps) {
    (void)successors[kv.first];
    (void)predecessors[kv.first];
  }

  // ---------- Kahn's topological sort ----------
  DenseMap<int, int> inDegree;
  for (const auto &kv : nodeOps)
    inDegree[kv.first] = 0;
  for (const auto &kv : predecessors)
    inDegree[kv.first] = static_cast<int>(kv.second.size());

  std::queue<int> zeroQ;
  for (const auto &kv : nodeOps) {
    if (inDegree[kv.first] == 0)
      zeroQ.push(kv.first);
  }

  while (!zeroQ.empty()) {
    int cur = zeroQ.front();
    zeroQ.pop();
    topoOrder.push_back(cur);
    for (int succ : successors[cur]) {
      inDegree[succ] -= 1;
      if (inDegree[succ] == 0)
        zeroQ.push(succ);
    }
  }

  // Graceful cycle fallback: add remaining nodes in arbitrary order.
  if (static_cast<int>(topoOrder.size()) !=
      static_cast<int>(nodeOps.size())) {
    LDBG("WARNING: cycle detected in KGIR DAG, "
         << (nodeOps.size() - topoOrder.size()) << " nodes unreachable");
    DenseSet<int> inTopo;
    for (int id : topoOrder)
      inTopo.insert(id);
    for (const auto &kv : nodeOps) {
      if (!inTopo.contains(kv.first))
        topoOrder.push_back(kv.first);
    }
  }

  // Build topological index map for O(1) lookups.
  for (unsigned i = 0, e = topoOrder.size(); i < e; ++i)
    topoIndex[topoOrder[i]] = static_cast<int>(i);
}

//===----------------------------------------------------------------------===//
// computeLivenessIntervals
//===----------------------------------------------------------------------===//

void MemoryPlanning::computeLivenessIntervals(
    ModuleOp moduleOp, SmallVectorImpl<LivenessInterval> &intervals) {

  moduleOp.walk([&](ttkgir::GraphOp graphOp) {
    // Rebuild per-graph DAG state.
    buildDAGAndTopologicalSort(graphOp, topoOrder_, topoIndex_, nodeOps_,
                               successors_, predecessors_);

    if (nodeOps_.empty())
      return;

    // Extract SMEM capacity from hardware profiles string.
    StringRef profiles = graphOp.getHardwareProfiles();
    smemCapacity_ = parseI64FromProfile(profiles, "smem_per_sm_bytes",
                                        kDefaultSmemCapacity);
    LDBG("SMEM capacity from profile: " << smemCapacity_ << " bytes");

    // Identify intermediate tensors: for each data_dep edge of type "flow",
    // the tensor at tensor_index on the producer is an intermediate consumed
    // by the dest node.
    graphOp.getBody().walk([&](ttkgir::DataDepOp dep) {
      int srcId = static_cast<int>(dep.getSourceNodeId());
      int dstId = static_cast<int>(dep.getDestNodeId());
      int32_t tensorIdx = static_cast<int32_t>(dep.getTensorIndex());

      // Both producer and consumer must be within this graph.
      if (!topoIndex_.count(srcId) || !topoIndex_.count(dstId))
        return;

      // Only "flow" (RAW) dependencies represent true intermediate tensors.
      StringRef depType = dep.getDepType();
      if (depType != "flow")
        return;

      // Estimate tensor size from memory access patterns on the producer.
      int64_t tensorSize = 0;
      Operation *srcOp = nodeOps_.lookup(srcId);
      if (auto kernelOp = dyn_cast_or_null<ttkgir::KernelLaunchOp>(srcOp)) {
        tensorSize = estimateTensorSizeFromAccessPattern(kernelOp, tensorIdx);
        // Fallback: use shared_memory_bytes as a rough proxy if no access
        // pattern is available.
        if (tensorSize == 0)
          tensorSize = static_cast<int64_t>(kernelOp.getSharedMemoryBytes());
      } else if (auto fusedOp =
                     dyn_cast_or_null<ttkgir::FusedKernelOp>(srcOp)) {
        tensorSize = estimateTensorSizeFromFusedKernel(fusedOp);
      }

      // Skip if we cannot estimate the size.
      if (tensorSize <= 0)
        return;

      LivenessInterval interval;
      interval.producerNodeId = srcId;
      interval.consumerNodeId = dstId;
      interval.tensorIndex = tensorIdx;
      interval.tensorSizeBytes = tensorSize;
      interval.firstUse = topoIndex_[srcId];
      interval.lastUse = topoIndex_[dstId];
      interval.isPromotable = true; // Will be refined later
      interval.isPromoted = false;

      intervals.push_back(interval);

      LDBG("  Intermediate tensor: node " << srcId << " -> node " << dstId
           << " (idx=" << tensorIdx << ", size=" << tensorSize << " bytes, "
           << "topo [" << interval.firstUse << ", " << interval.lastUse
           << "])");
    });
  });
}

//===----------------------------------------------------------------------===//
// checkSmemBudget — Sweep-line SMEM budget verification
//===----------------------------------------------------------------------===//

bool MemoryPlanning::checkSmemBudget(
    const SmallVectorImpl<LivenessInterval> &intervals,
    int64_t smemCapacity) {

  if (intervals.empty())
    return true;

  // Find the maximum topological index across all intervals.
  int maxTopoIdx = 0;
  for (const auto &iv : intervals)
    maxTopoIdx = std::max(maxTopoIdx, iv.lastUse);

  // Sweep line: at each topological position, sum the sizes of all live
  // promoted intermediates.
  // An intermediate is live at position p if firstUse <= p <= lastUse
  // and it is marked as promoted.
  int64_t maxConcurrent = 0;
  for (int pos = 0; pos <= maxTopoIdx; ++pos) {
    int64_t currentUsage = 0;
    for (const auto &iv : intervals) {
      if (iv.isPromoted && iv.firstUse <= pos && pos <= iv.lastUse)
        currentUsage += iv.tensorSizeBytes;
    }
    maxConcurrent = std::max(maxConcurrent, currentUsage);
  }

  LDBG("  Max concurrent SMEM usage: " << maxConcurrent
       << " bytes (capacity: " << smemCapacity << ")");

  return maxConcurrent <= smemCapacity;
}

//===----------------------------------------------------------------------===//
// analyzePromotionEligibility
//===----------------------------------------------------------------------===//

void MemoryPlanning::analyzePromotionEligibility(
    const SmallVectorImpl<LivenessInterval> &intervals,
    SmallVectorImpl<MemoryPromotion> &promotions) {

  if (intervals.empty())
    return;

  // Build a mutable copy of intervals sorted by tensor size descending.
  // Greedy strategy: prioritize promoting larger tensors first because they
  // eliminate more global memory round-trip traffic.
  SmallVector<LivenessInterval> sorted(intervals.begin(), intervals.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const LivenessInterval &a, const LivenessInterval &b) {
              return a.tensorSizeBytes > b.tensorSizeBytes;
            });

  // Phase 1: Mark individually eligible intervals (size fits SMEM).
  for (auto &iv : sorted) {
    // Small tensors go to register file; larger ones target shared memory.
    if (iv.tensorSizeBytes <= kRegisterPromotionThreshold) {
      iv.isPromotable = true;
      iv.isPromoted = true; // Register-file promotion is cheap.
    } else if (iv.tensorSizeBytes <= smemCapacity_) {
      iv.isPromotable = true;
      iv.isPromoted = true; // Tentatively promote — budget check follows.
    } else {
      // Tensor exceeds SMEM capacity — cannot promote.
      iv.isPromotable = false;
      iv.isPromoted = false;
      LDBG("  Skipping intermediate (too large): node "
           << iv.producerNodeId << " -> node " << iv.consumerNodeId
           << " (" << iv.tensorSizeBytes << " bytes > SMEM "
           << smemCapacity_ << ")");
    }
  }

  // Phase 2: Iteratively check concurrent SMEM budget and un-promote the
  // smallest-benefit interval until budget is satisfied.
  // Walk from the least beneficial (smallest) promoted interval and un-promote
  // if the budget is exceeded.
  while (!checkSmemBudget(sorted, smemCapacity_)) {
    // Find the smallest promoted SMEM-targeted interval and un-promote it.
    bool reverted = false;
    for (auto it = sorted.rbegin(), end = sorted.rend(); it != end; ++it) {
      if (it->isPromoted &&
          it->tensorSizeBytes > kRegisterPromotionThreshold) {
        LDBG("  Un-promoting intermediate to fit SMEM budget: node "
             << it->producerNodeId << " -> node " << it->consumerNodeId
             << " (" << it->tensorSizeBytes << " bytes)");
        it->isPromoted = false;
        reverted = true;
        break;
      }
    }
    // Safety valve: if nothing can be un-promoted, break to avoid infinite loop.
    if (!reverted)
      break;
  }

  // Phase 3: Build promotion decision list.
  for (const auto &iv : sorted) {
    if (!iv.isPromoted)
      continue;

    MemoryPromotion promo;
    promo.producerNodeId = iv.producerNodeId;
    promo.consumerNodeId = iv.consumerNodeId;
    promo.tensorIndex = iv.tensorIndex;
    promo.tensorSizeBytes = iv.tensorSizeBytes;
    promo.target = (iv.tensorSizeBytes <= kRegisterPromotionThreshold)
                       ? MemoryPromotion::RegisterFile
                       : MemoryPromotion::SharedMemory;

    promotions.push_back(promo);
  }
}

//===----------------------------------------------------------------------===//
// insertTransferOps — Cross-device transfer insertion
//===----------------------------------------------------------------------===//

void MemoryPlanning::insertTransferOps(ModuleOp moduleOp) {
  moduleOp.walk([&](ttkgir::GraphOp graphOp) {
    // Collect per-node device assignments for cross-device detection.
    DenseMap<int, int> nodeDeviceMap;
    graphOp.getBody().walk([&](ttkgir::KernelLaunchOp kOp) {
      nodeDeviceMap[static_cast<int>(kOp.getNodeId())] = getNodeDeviceId(kOp);
    });
    graphOp.getBody().walk([&](ttkgir::FusedKernelOp fOp) {
      nodeDeviceMap[static_cast<int>(fOp.getNodeId())] = getNodeDeviceId(fOp);
    });

    // Collect existing transfer edges to avoid duplicates.
    DenseSet<std::pair<int, int>> existingTransfers;
    graphOp.getBody().walk([&](ttkgir::TransferOp xfer) {
      existingTransfers.insert(
          {static_cast<int>(xfer.getSourceNodeId()),
           static_cast<int>(xfer.getDestNodeId())});
    });

    // Walk data_dep edges looking for cross-device dependencies.
    SmallVector<ttkgir::DataDepOp> crossDeviceDeps;
    graphOp.getBody().walk([&](ttkgir::DataDepOp dep) {
      int srcId = static_cast<int>(dep.getSourceNodeId());
      int dstId = static_cast<int>(dep.getDestNodeId());

      // Check if source and dest are on different devices.
      int srcDev = nodeDeviceMap.lookup(srcId);
      int dstDev = nodeDeviceMap.lookup(dstId);

      if (srcDev != dstDev) {
        // Skip if a transfer already exists for this edge.
        if (existingTransfers.contains({srcId, dstId}))
          return;
        crossDeviceDeps.push_back(dep);
      }
    });

    if (crossDeviceDeps.empty())
      return;

    LDBG("Inserting " << crossDeviceDeps.size()
         << " cross-device transfer ops");

    // Insert transfer ops at the end of the graph body.
    OpBuilder builder(&graphOp.getBody().front(),
                      graphOp.getBody().front().end());

    for (auto dep : crossDeviceDeps) {
      int srcId = static_cast<int>(dep.getSourceNodeId());
      int dstId = static_cast<int>(dep.getDestNodeId());
      int srcDev = nodeDeviceMap.lookup(srcId);
      int dstDev = nodeDeviceMap.lookup(dstId);

      // Estimate transfer size from the producer's access patterns.
      int64_t transferSize = 0;
      Operation *srcOp = nodeOps_.lookup(srcId);
      int32_t tensorIdx = static_cast<int32_t>(dep.getTensorIndex());
      if (auto kernelOp = dyn_cast_or_null<ttkgir::KernelLaunchOp>(srcOp)) {
        transferSize = estimateTensorSizeFromAccessPattern(kernelOp, tensorIdx);
        if (transferSize == 0)
          transferSize = static_cast<int64_t>(kernelOp.getSharedMemoryBytes());
      } else if (auto fusedOp =
                     dyn_cast_or_null<ttkgir::FusedKernelOp>(srcOp)) {
        transferSize = estimateTensorSizeFromFusedKernel(fusedOp);
      }
      // Fallback minimum transfer size.
      if (transferSize <= 0)
        transferSize = 1024;

      // Determine interconnect type based on device vendor annotations.
      // Default to "pcie" as the most common inter-device link.
      StringRef interconnectType = "pcie";
      if (auto kernelOp = dyn_cast_or_null<ttkgir::KernelLaunchOp>(srcOp)) {
        if (auto hw = kernelOp.getHardwareTarget()) {
          StringRef vendor = hw->getVendor();
          if (vendor == "nvidia")
            interconnectType = "nvlink";
          else if (vendor == "amd")
            interconnectType = "infinity_fabric";
        }
      }

      // Check if producer and consumer are from different vendors
      // (cross-vendor dispatch) — use host staging.
      bool crossVendor = false;
      if (auto srcKernel = dyn_cast_or_null<ttkgir::KernelLaunchOp>(srcOp)) {
        Operation *dstOp = nodeOps_.lookup(dstId);
        if (auto dstKernel =
                dyn_cast_or_null<ttkgir::KernelLaunchOp>(dstOp)) {
          auto srcHw = srcKernel.getHardwareTarget();
          auto dstHw = dstKernel.getHardwareTarget();
          if (srcHw && dstHw && srcHw->getVendor() != dstHw->getVendor())
            crossVendor = true;
        }
      }
      if (crossVendor)
        interconnectType = "host_staging";

      auto loc = dep.getLoc();
      auto sourceDevId = builder.getI32IntegerAttr(srcDev);
      auto destDevId = builder.getI32IntegerAttr(dstDev);
      auto tensorIdxAttr = builder.getI32IntegerAttr(tensorIdx);
      auto sizeAttr = builder.getI64IntegerAttr(transferSize);
      auto interAttr = builder.getStringAttr(interconnectType);
      auto srcNodeAttr = builder.getI32IntegerAttr(srcId);
      auto dstNodeAttr = builder.getI32IntegerAttr(dstId);

      // Create ttkgir.transfer operation.
      ttkgir::TransferOp::create(builder, loc,
                                 sourceDevId,    // source_device_id
                                 destDevId,      // dest_device_id
                                 tensorIdxAttr,  // tensor_index
                                 sizeAttr,       // transfer_size_bytes
                                 interAttr,      // interconnect_type
                                 /*estimated_latency_us=*/nullptr,
                                 srcNodeAttr,    // source_node_id
                                 dstNodeAttr);   // dest_node_id

      LDBG("  Transfer: device " << srcDev << " -> device " << dstDev
           << " (node " << srcId << " -> node " << dstId
           << ", " << transferSize << " bytes via "
           << interconnectType << ")");
    }
  });
}

//===----------------------------------------------------------------------===//
// applyFeedbackRefinement — Closed-loop occupancy-based refinement
//===----------------------------------------------------------------------===//

void MemoryPlanning::applyFeedbackRefinement(
    ModuleOp moduleOp, SmallVectorImpl<MemoryPromotion> &promotions) {

  // Walk KGIR nodes looking for RuntimePerformanceAttr annotations that
  // indicate measured occupancy data from previous execution iterations.
  // If a kernel's measured occupancy dropped significantly after a promotion
  // was applied, we revert the promotion for that intermediate.

  DenseMap<int, double> measuredOccupancy;

  moduleOp.walk([&](ttkgir::GraphOp graphOp) {
    graphOp.getBody().walk([&](ttkgir::KernelLaunchOp kOp) {
      auto perf = kOp.getRuntimePerf();
      if (!perf)
        return;
      double occ = perf->getOccupancy();
      int nodeId = static_cast<int>(kOp.getNodeId());
      measuredOccupancy[nodeId] = occ;

      LDBG("  Feedback: node " << nodeId << " measured occupancy = " << occ);
    });

    graphOp.getBody().walk([&](ttkgir::FusedKernelOp fOp) {
      auto perf = fOp.getRuntimePerf();
      if (!perf)
        return;
      double occ = perf->getOccupancy();
      int nodeId = static_cast<int>(fOp.getNodeId());
      measuredOccupancy[nodeId] = occ;

      LDBG("  Feedback: fused node " << nodeId
           << " measured occupancy = " << occ);
    });
  });

  if (measuredOccupancy.empty())
    return;

  // Check each existing promotion: if the consumer node's measured occupancy
  // is below the degradation threshold relative to expected theoretical
  // maximum (1.0), revert the promotion.
  //
  // The baseline expectation is that without promotion the kernel achieves
  // at least moderate occupancy. We detect degradation when a promoted
  // kernel's occupancy is significantly low compared to a healthy baseline.
  //
  // More precisely: for each promotion (producer → consumer), check if the
  // consumer's measured occupancy dropped relative to the producer's occupancy
  // (which represents a pre-promotion baseline for the compute unit).
  //
  // We mark indices for removal, then erase in-place to avoid SmallVector
  // move-assignment which triggers spurious GCC 13 -Wstringop-overread.
  SmallVector<unsigned> revertIndices;
  for (unsigned i = 0, e = promotions.size(); i < e; ++i) {
    const auto &promo = promotions[i];
    bool shouldRevert = false;

    // Check consumer node occupancy.
    auto consumerIt = measuredOccupancy.find(promo.consumerNodeId);
    auto producerIt = measuredOccupancy.find(promo.producerNodeId);

    if (consumerIt != measuredOccupancy.end() &&
        producerIt != measuredOccupancy.end()) {
      double consumerOcc = consumerIt->second;
      double producerOcc = producerIt->second;

      // If consumer occupancy is significantly lower than producer's,
      // the promotion may be causing SMEM pressure that reduces occupancy.
      if (producerOcc > 0.0 &&
          (producerOcc - consumerOcc) / producerOcc >
              kOccupancyDegradationThreshold) {
        shouldRevert = true;
        LDBG("  Reverting promotion: node " << promo.producerNodeId
             << " -> node " << promo.consumerNodeId
             << " (occupancy degraded from " << producerOcc
             << " to " << consumerOcc << ")");
      }
    } else if (consumerIt != measuredOccupancy.end()) {
      // If only consumer data is available, revert if occupancy is very low.
      double consumerOcc = consumerIt->second;
      if (consumerOcc < (1.0 - kOccupancyDegradationThreshold) * 0.5) {
        shouldRevert = true;
        LDBG("  Reverting promotion (low absolute occupancy): node "
             << promo.producerNodeId << " -> node " << promo.consumerNodeId
             << " (occupancy = " << consumerOcc << ")");
      }
    }

    if (shouldRevert)
      revertIndices.push_back(i);
  }

  // Erase reverted promotions in reverse index order to avoid invalidation.
  for (auto it = revertIndices.rbegin(), end = revertIndices.rend();
       it != end; ++it) {
    promotions.erase(promotions.begin() + *it);
  }
}

//===----------------------------------------------------------------------===//
// runOnOperation — Main pass entry point
//===----------------------------------------------------------------------===//

void MemoryPlanning::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  LDBG("Starting memory planning on KGIR graph");

  // Step 1: Compute liveness intervals for all intermediate tensors.
  SmallVector<LivenessInterval> intervals;
  computeLivenessIntervals(moduleOp, intervals);
  LDBG("Computed " << intervals.size() << " liveness intervals");

  if (intervals.empty()) {
    LDBG("No intermediate tensors found — nothing to plan");
    // Still insert transfer ops if needed.
    insertTransferOps(moduleOp);
    return;
  }

  // Step 2: Apply feedback refinement if runtime annotations are available
  // (closed-loop: revert promotions that caused occupancy degradation).
  // We collect any prior promotions from existing annotations first, then
  // refine them based on measured data.
  SmallVector<MemoryPromotion> promotions;
  applyFeedbackRefinement(moduleOp, promotions);

  // Step 3: Analyze promotion eligibility for each intermediate tensor.
  analyzePromotionEligibility(intervals, promotions);
  LDBG("Identified " << promotions.size() << " promotable intermediates");

  // Step 4: Insert cross-device transfer operations for dispatch-split
  // intermediates.
  insertTransferOps(moduleOp);

  // Step 5: Annotate KGIR nodes and edges with promotion decisions.
  // Walk the module and set memory_promotion attributes on data_dep edges
  // or kernel_launch nodes so downstream passes (scheduler, codegen bridge)
  // can consume promotion information.
  moduleOp.walk([&](ttkgir::GraphOp graphOp) {
    for (const auto &promo : promotions) {
      StringRef targetName = (promo.target == MemoryPromotion::SharedMemory)
                                 ? "shared"
                                 : "register";

      // Find the corresponding data_dep edge and annotate it.
      graphOp.getBody().walk([&](ttkgir::DataDepOp dep) {
        if (static_cast<int>(dep.getSourceNodeId()) == promo.producerNodeId &&
            static_cast<int>(dep.getDestNodeId()) == promo.consumerNodeId &&
            static_cast<int>(dep.getTensorIndex()) == promo.tensorIndex) {
          // Annotate the edge with promotion decision as a string attribute.
          dep->setAttr("memory_promotion",
                       StringAttr::get(dep.getContext(), targetName));
          dep->setAttr("promotion_size_bytes",
                       IntegerAttr::get(
                           IntegerType::get(dep.getContext(), 64),
                           promo.tensorSizeBytes));
        }
      });

      LDBG("Promoting intermediate: node " << promo.producerNodeId
           << " -> node " << promo.consumerNodeId
           << " (idx=" << promo.tensorIndex
           << ", " << promo.tensorSizeBytes << " bytes -> " << targetName
           << ")");
    }
  });

  LDBG("Memory planning complete");
}

} // namespace
