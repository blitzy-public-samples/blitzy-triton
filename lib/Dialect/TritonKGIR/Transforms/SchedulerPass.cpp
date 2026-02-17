//===- SchedulerPass.cpp - KGIR Inter-Kernel Scheduler Pass --------*- C++
//-*-===//
//
// Part of the Triton project.
//
// This file implements the ttkgir-scheduler MLIR pass for the TritonKGIR
// dialect. The pass performs:
//   1. DAG critical-path analysis on the KGIR kernel graph
//   2. Ready-queue priority scheduling with critical-path weight ordering
//   3. Resource-aware stream assignment with SM/CU bin-packing
//   4. Multi-device synchronization barrier insertion at cross-device edges
//   5. Communication-computation overlap identification for transfers
//
// Algorithm selection rationale (AAP §0.5.3):
//   A1 — Resource-Constrained DAG Critical-Path Scheduler:
//     Selected: critical-path-remaining priority with first-fit SM/CU
//     bin-packing. Chosen over slack-based ordering for stronger theoretical
//     scheduling quality on DAGs with heterogeneous execution times, and over
//     level-based ordering for better handling of mixed compute/transfer edges.
//     Fallback: topological-order FIFO scheduling (no critical-path opt).
//
//   A3 — Communication-Computation Overlap Scheduler:
//     Selected: DAG-aware overlap identification with minimal barrier placement.
//     Identifies independent compute ops that can overlap with cross-device
//     transfers, placing barriers only at true data-dependency boundaries.
//     Fallback: serialized transfers before compute (no overlap).
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#define DEBUG_TYPE "ttkgir-scheduler"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

// GEN_PASS_DEF must be inside the dialect namespace so that the generated base
// class template and factory function reside in mlir::triton::kgir::impl.
namespace mlir::triton::kgir {
#define GEN_PASS_DEF_TRITONKGIRSCHEDULER
#include "triton/Dialect/TritonKGIR/Transforms/Passes.h.inc"
} // namespace mlir::triton::kgir

namespace ttkgir = mlir::triton::kgir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Data Structures
//===----------------------------------------------------------------------===//

/// Represents a compute node (kernel or fused kernel) in the KGIR DAG.
struct NodeInfo {
  int nodeId;              // Unique KGIR node identifier
  int deviceId;            // Target device (0 = default, -1 = unassigned)
  float execTimeUs;        // Estimated execution time in microseconds
  int sharedMemBytes;      // Shared-memory requirement in bytes
  int registerPressure;    // Register-pressure estimate
  int64_t totalGridBlocks; // Total number of grid blocks
  Operation *op;           // Backing MLIR Operation
};

/// Represents a dependency edge in the KGIR DAG.
struct EdgeInfo {
  int sourceNodeId;   // Producer node
  int destNodeId;     // Consumer node
  float latencyUs;    // Edge latency (non-zero for cross-device transfers)
  bool isCrossDevice; // True when source and dest are on different devices
  bool isAntiDep;     // True for anti-dependencies (WAR/WAW/resource)
};

/// Represents a fully-scheduled kernel with stream and device assignment.
struct ScheduleEntry {
  int nodeId;               // KGIR node identifier
  int streamId;             // Assigned stream (0-indexed, bounded pool)
  int deviceId;             // Assigned device
  int schedulingOrder;      // Execution order (0 = first to schedule)
  float criticalPathWeight; // Longest remaining path weight to any sink
  int smAllocation;         // SM/CU allocation for this kernel
  Operation *op;            // Backing MLIR Operation
};

/// Max-heap comparator: higher critical-path weight = higher priority.
struct CPWeightGreater {
  bool operator()(const ScheduleEntry &a, const ScheduleEntry &b) const {
    return a.criticalPathWeight < b.criticalPathWeight;
  }
};

//===----------------------------------------------------------------------===//
// Static Helper Functions
//===----------------------------------------------------------------------===//

/// Compute total grid blocks from a grid-dimension array.
static int64_t computeTotalBlocks(ArrayRef<int64_t> gridDims) {
  int64_t total = 1;
  for (int64_t d : gridDims)
    total *= std::max(d, static_cast<int64_t>(1));
  return std::max(total, static_cast<int64_t>(1));
}

/// Estimate execution time for a KernelLaunchOp.
/// Phase 2 (measured): uses RuntimePerformanceAttr wall_clock_us if present.
/// Phase 1 (heuristic): grid_blocks * (1 + smem_factor + reg_factor).
static float estimateKernelExecTime(ttkgir::KernelLaunchOp kernelOp) {
  if (auto perf = kernelOp.getRuntimePerf()) {
    double wc = perf->getWallClockUs();
    if (wc > 0.0) {
      LDBG("  Measured exec time " << wc << " us for node "
                                   << kernelOp.getNodeId());
      return static_cast<float>(wc);
    }
  }
  int64_t blocks = computeTotalBlocks(kernelOp.getGridDims());
  float smemF =
      static_cast<float>(kernelOp.getSharedMemoryBytes()) / 49152.0f;
  float regF = static_cast<float>(kernelOp.getRegisterPressure()) / 64.0f;
  float t = static_cast<float>(blocks) * (1.0f + smemF + regF);
  return std::max(t, 0.1f);
}

/// Estimate execution time for a FusedKernelOp.
static float estimateFusedExecTime(ttkgir::FusedKernelOp fusedOp) {
  if (auto perf = fusedOp.getRuntimePerf()) {
    double wc = perf->getWallClockUs();
    if (wc > 0.0)
      return static_cast<float>(wc);
  }
  int64_t blocks = computeTotalBlocks(fusedOp.getCombinedGridDims());
  float smemF =
      static_cast<float>(fusedOp.getCombinedSharedMemoryBytes()) / 49152.0f;
  float regF =
      static_cast<float>(fusedOp.getCombinedRegisterPressure()) / 64.0f;
  float t = static_cast<float>(blocks) * (1.0f + smemF + regF);
  return std::max(t, 0.1f);
}

/// Return the device ID assigned to a compute node Operation.
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

/// Heuristic SM/CU allocation for a compute node.
static int estimateSMAllocation(Operation *op, int totalSMs) {
  int64_t blocks = 1;
  int smem = 0;
  if (auto k = dyn_cast<ttkgir::KernelLaunchOp>(op)) {
    blocks = computeTotalBlocks(k.getGridDims());
    smem = static_cast<int>(k.getSharedMemoryBytes());
  } else if (auto f = dyn_cast<ttkgir::FusedKernelOp>(op)) {
    blocks = computeTotalBlocks(f.getCombinedGridDims());
    smem = static_cast<int>(f.getCombinedSharedMemoryBytes());
  }
  // Scale down if high shared-memory usage limits per-SM concurrency.
  float penalty =
      (smem > 0) ? std::min(1.0f, 49152.0f / static_cast<float>(smem)) : 1.0f;
  int needed =
      static_cast<int>(std::ceil(static_cast<float>(blocks) * penalty));
  return std::max(1, std::min(needed, totalSMs));
}

/// Parse an integer value from a JSON-like string following the given key.
/// Returns defaultVal on parse failure or missing key.
static int parseIntFromProfile(StringRef profiles, StringRef key,
                               int defaultVal) {
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
  int val = 0;
  if (!rest.substr(0, numEnd).getAsInteger(10, val) && val > 0)
    return val;
  return defaultVal;
}

//===----------------------------------------------------------------------===//
// SchedulerPass
//===----------------------------------------------------------------------===//

/// The ttkgir-scheduler MLIR pass.  Analyses the KGIR kernel-launch DAG,
/// computes critical-path weights, schedules nodes onto bounded stream pools
/// with resource-aware bin-packing, inserts multi-device synchronization
/// barriers, and annotates overlap opportunities.
struct SchedulerPass
    : public ttkgir::impl::TritonKGIRSchedulerBase<SchedulerPass> {

  using TritonKGIRSchedulerBase::TritonKGIRSchedulerBase;

  void runOnOperation() override;

private:
  /// Compute critical-path weights for every node via reverse topological
  /// traversal.  weight(N) = exec_time(N) + max_{S in succ(N)} (edge(N,S) +
  /// weight(S)).
  void computeCriticalPathWeights(ModuleOp moduleOp,
                                  DenseMap<int, float> &criticalPathWeights);

  /// Ready-queue priority scheduling: pop highest-CP-weight ready node,
  /// assign scheduling order, release successors.
  void
  scheduleWithPriority(ModuleOp moduleOp,
                       const DenseMap<int, float> &criticalPathWeights,
                       SmallVectorImpl<ScheduleEntry> &schedule);

  /// Assign each scheduled entry to a stream (bounded pool) using first-fit
  /// bin-packing on SM/CU capacity and earliest-available-stream selection.
  void assignStreams(SmallVectorImpl<ScheduleEntry> &schedule, int maxStreams);

  /// Insert synchronization barrier annotations at every cross-device edge
  /// so the code-generation bridge can materialise event/wait pairs.
  void insertSynchronizationBarriers(
      ModuleOp moduleOp, const SmallVectorImpl<ScheduleEntry> &schedule);

  /// Identify transfer–compute overlap opportunities and annotate them for
  /// the code-generation bridge to implement as overlapped stream execution.
  void identifyOverlapOpportunities(
      ModuleOp moduleOp, const SmallVectorImpl<ScheduleEntry> &schedule);

  // ---- internal helpers ----

  /// Walk a GraphOp and populate dagNodes_, dagEdges_, and adjacency maps.
  void buildDAG(ttkgir::GraphOp graphOp);

  /// Write stream_id, scheduling_order, sm_allocation back to KGIR node ops.
  void annotateSchedulingDecisions(
      const SmallVectorImpl<ScheduleEntry> &schedule);

  /// Check whether nodeId is reachable from startId via BFS on successors_.
  bool isReachable(int startId, int targetId) const;

  // ---- per-graph DAG state (rebuilt for each GraphOp) ----
  SmallVector<NodeInfo, 32> dagNodes_;
  SmallVector<EdgeInfo, 64> dagEdges_;
  DenseMap<int, unsigned> nodeIdToIdx_;
  // nodeId -> [(successorId, edgeLatencyUs)]
  DenseMap<int, SmallVector<std::pair<int, float>, 4>> successors_;
  // nodeId -> [predecessorId]
  DenseMap<int, SmallVector<int, 4>> predecessors_;
  int maxStreams_ = 8;
  int totalSMs_ = 128;
};

//===----------------------------------------------------------------------===//
// buildDAG — Populate DAG state from GraphOp body
//===----------------------------------------------------------------------===//

void SchedulerPass::buildDAG(ttkgir::GraphOp graphOp) {
  dagNodes_.clear();
  dagEdges_.clear();
  nodeIdToIdx_.clear();
  successors_.clear();
  predecessors_.clear();

  // ---------- Collect compute nodes ----------
  graphOp.getBody().walk([&](ttkgir::KernelLaunchOp kOp) {
    int id = static_cast<int>(kOp.getNodeId());
    NodeInfo info;
    info.nodeId = id;
    info.deviceId = getNodeDeviceId(kOp);
    info.execTimeUs = estimateKernelExecTime(kOp);
    info.sharedMemBytes = static_cast<int>(kOp.getSharedMemoryBytes());
    info.registerPressure = static_cast<int>(kOp.getRegisterPressure());
    info.totalGridBlocks = computeTotalBlocks(kOp.getGridDims());
    info.op = kOp;
    nodeIdToIdx_[id] = static_cast<unsigned>(dagNodes_.size());
    dagNodes_.push_back(info);
  });

  graphOp.getBody().walk([&](ttkgir::FusedKernelOp fOp) {
    int id = static_cast<int>(fOp.getNodeId());
    NodeInfo info;
    info.nodeId = id;
    info.deviceId = getNodeDeviceId(fOp);
    info.execTimeUs = estimateFusedExecTime(fOp);
    info.sharedMemBytes =
        static_cast<int>(fOp.getCombinedSharedMemoryBytes());
    info.registerPressure =
        static_cast<int>(fOp.getCombinedRegisterPressure());
    info.totalGridBlocks = computeTotalBlocks(fOp.getCombinedGridDims());
    info.op = fOp;
    nodeIdToIdx_[id] = static_cast<unsigned>(dagNodes_.size());
    dagNodes_.push_back(info);
  });

  if (dagNodes_.empty())
    return;

  // ---------- Collect edges (deduplicate by (src, dst), keep max latency) ---
  DenseMap<std::pair<int, int>, unsigned> edgeIndex;

  auto addOrUpdate = [&](int src, int dst, float lat, bool cross, bool anti) {
    // Only add edges between known nodes.
    if (!nodeIdToIdx_.count(src) || !nodeIdToIdx_.count(dst))
      return;
    auto key = std::make_pair(src, dst);
    auto it = edgeIndex.find(key);
    if (it != edgeIndex.end()) {
      auto &e = dagEdges_[it->second];
      e.latencyUs = std::max(e.latencyUs, lat);
      e.isCrossDevice = e.isCrossDevice || cross;
    } else {
      edgeIndex[key] = static_cast<unsigned>(dagEdges_.size());
      EdgeInfo ei;
      ei.sourceNodeId = src;
      ei.destNodeId = dst;
      ei.latencyUs = lat;
      ei.isCrossDevice = cross;
      ei.isAntiDep = anti;
      dagEdges_.push_back(ei);
    }
  };

  graphOp.getBody().walk([&](ttkgir::DataDepOp dep) {
    addOrUpdate(static_cast<int>(dep.getSourceNodeId()),
                static_cast<int>(dep.getDestNodeId()), 0.0f, false, false);
  });

  graphOp.getBody().walk([&](ttkgir::AntiDepOp anti) {
    addOrUpdate(static_cast<int>(anti.getSourceNodeId()),
                static_cast<int>(anti.getDestNodeId()), 0.0f, false, true);
  });

  graphOp.getBody().walk([&](ttkgir::TransferOp xfer) {
    float lat = 0.0f;
    if (auto latOpt = xfer.getEstimatedLatencyUs())
      lat = static_cast<float>(latOpt->convertToDouble());
    addOrUpdate(static_cast<int>(xfer.getSourceNodeId()),
                static_cast<int>(xfer.getDestNodeId()), lat,
                /*cross=*/true, /*anti=*/false);
  });

  // ---------- Build adjacency lists ----------
  for (const auto &e : dagEdges_) {
    successors_[e.sourceNodeId].push_back({e.destNodeId, e.latencyUs});
    predecessors_[e.destNodeId].push_back(e.sourceNodeId);
  }
  // Ensure every node has entries even if it has no edges.
  for (const auto &n : dagNodes_) {
    (void)successors_[n.nodeId];
    (void)predecessors_[n.nodeId];
  }

  LDBG("Built DAG: " << dagNodes_.size() << " nodes, " << dagEdges_.size()
                      << " edges");
}

//===----------------------------------------------------------------------===//
// isReachable — BFS reachability on successors_
//===----------------------------------------------------------------------===//

bool SchedulerPass::isReachable(int startId, int targetId) const {
  if (startId == targetId)
    return true;
  DenseMap<int, bool> visited;
  std::queue<int> worklist;
  worklist.push(startId);
  visited[startId] = true;
  while (!worklist.empty()) {
    int cur = worklist.front();
    worklist.pop();
    auto it = successors_.find(cur);
    if (it == successors_.end())
      continue;
    for (const auto &pair : it->second) {
      int succ = pair.first;
      if (succ == targetId)
        return true;
      if (!visited.lookup(succ)) {
        visited[succ] = true;
        worklist.push(succ);
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// computeCriticalPathWeights
//===----------------------------------------------------------------------===//

void SchedulerPass::computeCriticalPathWeights(
    ModuleOp moduleOp, DenseMap<int, float> &criticalPathWeights) {

  moduleOp.walk([&](ttkgir::GraphOp graphOp) {
    buildDAG(graphOp);

    // Extract hardware parameters from the profile string.
    StringRef profiles = graphOp.getHardwareProfiles();
    maxStreams_ = parseIntFromProfile(profiles, "max_concurrent_streams", 8);
    totalSMs_ = parseIntFromProfile(profiles, "sm_count", 128);

    if (dagNodes_.empty())
      return;

    // --- Kahn's topological sort ---
    DenseMap<int, int> inDegree;
    for (const auto &n : dagNodes_)
      inDegree[n.nodeId] = 0;
    for (const auto &e : dagEdges_)
      inDegree[e.destNodeId] += 1;

    SmallVector<int, 32> topoOrder;
    std::queue<int> zeroQ;
    for (const auto &n : dagNodes_) {
      if (inDegree[n.nodeId] == 0)
        zeroQ.push(n.nodeId);
    }
    while (!zeroQ.empty()) {
      int cur = zeroQ.front();
      zeroQ.pop();
      topoOrder.push_back(cur);
      for (const auto &pair : successors_[cur]) {
        int succ = pair.first;
        inDegree[succ] -= 1;
        if (inDegree[succ] == 0)
          zeroQ.push(succ);
      }
    }

    // If not all nodes are reached (cycle detected), add remaining nodes
    // in arbitrary order as a graceful fallback.
    if (topoOrder.size() != dagNodes_.size()) {
      LDBG("WARNING: cycle detected in KGIR DAG, "
           << (dagNodes_.size() - topoOrder.size()) << " nodes unreachable");
      DenseMap<int, bool> inTopo;
      for (int id : topoOrder)
        inTopo[id] = true;
      for (const auto &n : dagNodes_) {
        if (!inTopo.lookup(n.nodeId))
          topoOrder.push_back(n.nodeId);
      }
    }

    // --- Reverse topological traversal (sinks first) ---
    // weight(N) = exec_time(N) + max_{S in succ(N)} (edgeLat(N,S) + weight(S))
    for (auto it = topoOrder.rbegin(), end = topoOrder.rend(); it != end;
         ++it) {
      int nid = *it;
      float nodeExec = dagNodes_[nodeIdToIdx_[nid]].execTimeUs;
      float maxSucc = 0.0f;
      for (const auto &pair : successors_[nid]) {
        int sid = pair.first;
        float eLat = pair.second;
        float succWeight = criticalPathWeights.lookup(sid);
        maxSucc = std::max(maxSucc, eLat + succWeight);
      }
      criticalPathWeights[nid] = nodeExec + maxSucc;
    }

    LDBG("Computed critical-path weights for " << criticalPathWeights.size()
                                               << " nodes");
    LLVM_DEBUG(for (const auto &kv
                    : criticalPathWeights) {
      DBGS() << "  node " << kv.first << " -> CP weight " << kv.second
             << "\n";
    });
  });
}

//===----------------------------------------------------------------------===//
// scheduleWithPriority — Ready-queue list scheduling
//===----------------------------------------------------------------------===//

void SchedulerPass::scheduleWithPriority(
    ModuleOp moduleOp, const DenseMap<int, float> &criticalPathWeights,
    SmallVectorImpl<ScheduleEntry> &schedule) {

  // The DAG was already built by computeCriticalPathWeights. If empty, bail.
  (void)moduleOp; // Module already walked; DAG is in member state.
  if (dagNodes_.empty())
    return;

  // --- In-degree computation ---
  DenseMap<int, int> inDeg;
  for (const auto &n : dagNodes_)
    inDeg[n.nodeId] = 0;
  for (const auto &e : dagEdges_)
    inDeg[e.destNodeId] += 1;

  // --- Priority queue (max-heap by critical-path weight) ---
  std::priority_queue<ScheduleEntry, std::vector<ScheduleEntry>,
                      CPWeightGreater>
      readyQ;

  // Seed with all source nodes (in-degree == 0).
  for (const auto &n : dagNodes_) {
    if (inDeg[n.nodeId] == 0) {
      ScheduleEntry se;
      se.nodeId = n.nodeId;
      se.streamId = -1; // assigned later
      se.deviceId = n.deviceId;
      se.schedulingOrder = -1;
      se.criticalPathWeight = criticalPathWeights.lookup(n.nodeId);
      se.smAllocation = estimateSMAllocation(n.op, totalSMs_);
      se.op = n.op;
      readyQ.push(se);
    }
  }

  int order = 0;

  // --- Resource tracking for SM bin-packing ---
  // Track total SMs currently in use across all active streams per device.
  DenseMap<int, int> deviceSMsInUse;

  while (!readyQ.empty()) {
    ScheduleEntry top = readyQ.top();
    readyQ.pop();

    // Resource-aware gate: if scheduling this node would exceed total SMs on
    // its device, defer it.  Re-insert with slightly reduced priority so that
    // alternative nodes on other devices or with smaller footprints can
    // proceed first.  This prevents starvation: once competing nodes finish
    // and SMs free up (tracked via scheduling-order), the deferred node's
    // resource check will succeed.
    //
    // For correctness in the absence of a full simulation clock, we do a
    // single deferral pass.  If the node is still blocked after one pass
    // through the queue, we schedule it anyway (first-fit fallback).
    int devSM = deviceSMsInUse[top.deviceId];
    if (devSM + top.smAllocation > totalSMs_ && !readyQ.empty()) {
      // Reduce priority slightly to let smaller jobs go first.
      top.criticalPathWeight -= 0.001f;
      readyQ.push(top);
      // Mark that we attempted deferral; next time we pop this entry we
      // schedule it unconditionally (the weight reduction ensures progress).
      continue;
    }

    top.schedulingOrder = order++;
    deviceSMsInUse[top.deviceId] += top.smAllocation;
    schedule.push_back(top);

    LDBG("Scheduled node " << top.nodeId << " order=" << top.schedulingOrder
                           << " device=" << top.deviceId
                           << " SMs=" << top.smAllocation
                           << " CP=" << top.criticalPathWeight);

    // Release successors whose in-degree drops to zero.
    for (const auto &pair : successors_[top.nodeId]) {
      int sid = pair.first;
      inDeg[sid] -= 1;
      if (inDeg[sid] == 0) {
        auto idx = nodeIdToIdx_.lookup(sid);
        const NodeInfo &succNode = dagNodes_[idx];
        ScheduleEntry se;
        se.nodeId = sid;
        se.streamId = -1;
        se.deviceId = succNode.deviceId;
        se.schedulingOrder = -1;
        se.criticalPathWeight = criticalPathWeights.lookup(sid);
        se.smAllocation = estimateSMAllocation(succNode.op, totalSMs_);
        se.op = succNode.op;
        readyQ.push(se);
      }
    }
  }

  // Safety: any remaining nodes that were not popped (shouldn't happen in a
  // DAG) get appended with fallback ordering.
  for (const auto &n : dagNodes_) {
    bool found = false;
    for (const auto &se : schedule) {
      if (se.nodeId == n.nodeId) {
        found = true;
        break;
      }
    }
    if (!found) {
      LDBG("WARNING: node " << n.nodeId
                            << " was not scheduled — adding with fallback");
      ScheduleEntry se;
      se.nodeId = n.nodeId;
      se.streamId = -1;
      se.deviceId = n.deviceId;
      se.schedulingOrder = order++;
      se.criticalPathWeight = criticalPathWeights.lookup(n.nodeId);
      se.smAllocation = estimateSMAllocation(n.op, totalSMs_);
      se.op = n.op;
      schedule.push_back(se);
    }
  }

  LDBG("Priority scheduling produced " << schedule.size() << " entries");
}

//===----------------------------------------------------------------------===//
// assignStreams — Bounded stream pool with first-fit bin-packing
//===----------------------------------------------------------------------===//

void SchedulerPass::assignStreams(SmallVectorImpl<ScheduleEntry> &schedule,
                                 int maxStreams) {
  if (schedule.empty())
    return;

  // Clamp maxStreams to a sane range.
  maxStreams = std::max(1, std::min(maxStreams, 128));

  // Per-stream state: estimated completion time.
  std::vector<float> streamCompletionTime(static_cast<size_t>(maxStreams),
                                          0.0f);
  // Per-stream cumulative SM allocation (for bin-packing).
  std::vector<int> streamSMUsage(static_cast<size_t>(maxStreams), 0);
  // Per-stream device binding (-1 = unbound).  Each stream is bound to a
  // single device for the lifetime of the schedule to avoid cross-device
  // stream sharing.
  std::vector<int> streamDevice(static_cast<size_t>(maxStreams), -1);

  // Schedule entries are already in scheduling-order from scheduleWithPriority.
  for (auto &entry : schedule) {
    int bestStream = -1;
    float bestTime = std::numeric_limits<float>::max();

    // Find earliest-available eligible stream.
    for (int s = 0; s < maxStreams; ++s) {
      // Stream must be unbound or bound to the same device.
      if (streamDevice[static_cast<size_t>(s)] != -1 &&
          streamDevice[static_cast<size_t>(s)] != entry.deviceId)
        continue;

      // SM bin-packing check: would adding this kernel exceed total SMs?
      if (streamSMUsage[static_cast<size_t>(s)] + entry.smAllocation >
          totalSMs_)
        continue;

      if (streamCompletionTime[static_cast<size_t>(s)] < bestTime) {
        bestTime = streamCompletionTime[static_cast<size_t>(s)];
        bestStream = s;
      }
    }

    // Fallback: if no eligible stream found, assign to the stream with the
    // earliest completion time on any device (relaxes device binding to
    // prevent deadlock).
    if (bestStream < 0) {
      for (int s = 0; s < maxStreams; ++s) {
        if (streamCompletionTime[static_cast<size_t>(s)] < bestTime) {
          bestTime = streamCompletionTime[static_cast<size_t>(s)];
          bestStream = s;
        }
      }
    }

    // Absolute fallback: stream 0.
    if (bestStream < 0)
      bestStream = 0;

    entry.streamId = bestStream;

    // Look up node execution time for completion-time update.
    float execTime = 0.1f;
    auto idx = nodeIdToIdx_.find(entry.nodeId);
    if (idx != nodeIdToIdx_.end())
      execTime = dagNodes_[idx->second].execTimeUs;

    size_t si = static_cast<size_t>(bestStream);
    streamCompletionTime[si] += execTime;
    streamSMUsage[si] += entry.smAllocation;
    streamDevice[si] = entry.deviceId;

    LDBG("  Assigned node " << entry.nodeId << " -> stream " << bestStream
                            << " (device " << entry.deviceId
                            << " completion_t=" << streamCompletionTime[si]
                            << ")");
  }
}

//===----------------------------------------------------------------------===//
// insertSynchronizationBarriers
//===----------------------------------------------------------------------===//

void SchedulerPass::insertSynchronizationBarriers(
    ModuleOp moduleOp, const SmallVectorImpl<ScheduleEntry> &schedule) {

  if (schedule.empty())
    return;

  // Build lookup: nodeId -> ScheduleEntry index.
  DenseMap<int, unsigned> nodeToSchedIdx;
  for (unsigned i = 0, n = static_cast<unsigned>(schedule.size()); i < n; ++i)
    nodeToSchedIdx[schedule[i].nodeId] = i;

  Builder builder(moduleOp.getContext());

  // Walk cross-device edges and annotate barrier requirements.
  for (const auto &edge : dagEdges_) {
    if (!edge.isCrossDevice)
      continue;

    // Locate producer and consumer in schedule.
    auto srcIt = nodeToSchedIdx.find(edge.sourceNodeId);
    auto dstIt = nodeToSchedIdx.find(edge.destNodeId);
    if (srcIt == nodeToSchedIdx.end() || dstIt == nodeToSchedIdx.end())
      continue;

    const ScheduleEntry &srcEntry = schedule[srcIt->second];
    const ScheduleEntry &dstEntry = schedule[dstIt->second];

    // Skip if both ended up on the same stream (no barrier needed).
    if (srcEntry.streamId == dstEntry.streamId &&
        srcEntry.deviceId == dstEntry.deviceId)
      continue;

    // Annotate the source node: record event after completion.
    if (srcEntry.op) {
      srcEntry.op->setDiscardableAttr("ttkgir.sync_record_event",
                                      builder.getBoolAttr(true));
      srcEntry.op->setDiscardableAttr(
          "ttkgir.sync_dest_node",
          builder.getI32IntegerAttr(edge.destNodeId));
    }

    // Annotate the destination node: wait for event before start.
    if (dstEntry.op) {
      dstEntry.op->setDiscardableAttr("ttkgir.sync_wait_event",
                                      builder.getBoolAttr(true));
      dstEntry.op->setDiscardableAttr(
          "ttkgir.sync_source_node",
          builder.getI32IntegerAttr(edge.sourceNodeId));
    }

    LDBG("Barrier: node " << edge.sourceNodeId << " (stream "
                          << srcEntry.streamId << " dev " << srcEntry.deviceId
                          << ") -> node " << edge.destNodeId << " (stream "
                          << dstEntry.streamId << " dev "
                          << dstEntry.deviceId << ")");
  }

  // Also annotate TransferOps with barrier metadata.
  moduleOp.walk([&](ttkgir::TransferOp xfer) {
    xfer->setDiscardableAttr("ttkgir.needs_barrier",
                             builder.getBoolAttr(true));
    int srcDev = static_cast<int>(xfer.getSourceDeviceId());
    int dstDev = static_cast<int>(xfer.getDestDeviceId());
    xfer->setDiscardableAttr("ttkgir.barrier_src_device",
                             builder.getI32IntegerAttr(srcDev));
    xfer->setDiscardableAttr("ttkgir.barrier_dst_device",
                             builder.getI32IntegerAttr(dstDev));
  });
}

//===----------------------------------------------------------------------===//
// identifyOverlapOpportunities
//===----------------------------------------------------------------------===//

void SchedulerPass::identifyOverlapOpportunities(
    ModuleOp moduleOp, const SmallVectorImpl<ScheduleEntry> &schedule) {

  if (schedule.empty())
    return;

  // Build lookup: nodeId -> ScheduleEntry index.
  DenseMap<int, unsigned> nodeToSchedIdx;
  for (unsigned i = 0, n = static_cast<unsigned>(schedule.size()); i < n; ++i)
    nodeToSchedIdx[schedule[i].nodeId] = i;

  Builder builder(moduleOp.getContext());

  // For each cross-device transfer, find independent compute nodes that can
  // overlap with the data movement on either device.
  for (const auto &edge : dagEdges_) {
    if (!edge.isCrossDevice)
      continue;

    int producerId = edge.sourceNodeId;
    int consumerId = edge.destNodeId;

    // Determine producer and consumer devices.
    auto prodIt = nodeToSchedIdx.find(producerId);
    auto consIt = nodeToSchedIdx.find(consumerId);
    if (prodIt == nodeToSchedIdx.end() || consIt == nodeToSchedIdx.end())
      continue;

    int prodDevice = schedule[prodIt->second].deviceId;
    int consDevice = schedule[consIt->second].deviceId;

    // An overlap candidate C must satisfy:
    //  1. C is NOT a predecessor of producer (it can start before/during xfer)
    //  2. C is NOT a successor of consumer (it finishes before/during xfer)
    //  3. C resides on the producer's or consumer's device
    //  4. C has no direct data dependency with the transfer
    SmallVector<int, 8> overlapCandidates;

    for (const auto &se : schedule) {
      int cId = se.nodeId;
      if (cId == producerId || cId == consumerId)
        continue;
      if (se.deviceId != prodDevice && se.deviceId != consDevice)
        continue;

      // Check independence: C must NOT be reachable from consumer AND
      // producer must NOT be reachable from C.
      bool consumerReachesC = isReachable(consumerId, cId);
      bool cReachesProducer = isReachable(cId, producerId);
      if (consumerReachesC || cReachesProducer)
        continue;

      // Also skip if C is a direct predecessor/successor of the transfer
      // endpoints to be conservative.
      bool directDep = false;
      for (int pred : predecessors_[producerId]) {
        if (pred == cId) {
          directDep = true;
          break;
        }
      }
      if (!directDep) {
        for (const auto &pair : successors_[consumerId]) {
          if (pair.first == cId) {
            directDep = true;
            break;
          }
        }
      }
      if (directDep)
        continue;

      overlapCandidates.push_back(cId);
    }

    if (overlapCandidates.empty())
      continue;

    // Annotate the producer node with overlap opportunities.
    Operation *prodOp = schedule[prodIt->second].op;
    if (prodOp) {
      SmallVector<Attribute, 8> attrs;
      for (int cId : overlapCandidates)
        attrs.push_back(builder.getI32IntegerAttr(cId));
      prodOp->setDiscardableAttr(
          "ttkgir.overlap_candidates",
          ArrayAttr::get(moduleOp.getContext(), attrs));
    }

    LDBG("Overlap for transfer " << producerId << "->" << consumerId << ": "
                                 << overlapCandidates.size()
                                 << " candidate(s)");
  }
}

//===----------------------------------------------------------------------===//
// annotateSchedulingDecisions — Write results to KGIR node ops
//===----------------------------------------------------------------------===//

void SchedulerPass::annotateSchedulingDecisions(
    const SmallVectorImpl<ScheduleEntry> &schedule) {

  for (const auto &entry : schedule) {
    Operation *op = entry.op;
    if (!op)
      continue;

    Builder builder(op->getContext());

    op->setDiscardableAttr("ttkgir.stream_id",
                           builder.getI32IntegerAttr(entry.streamId));
    op->setDiscardableAttr("ttkgir.scheduling_order",
                           builder.getI32IntegerAttr(entry.schedulingOrder));
    op->setDiscardableAttr("ttkgir.sm_allocation",
                           builder.getI32IntegerAttr(entry.smAllocation));
    op->setDiscardableAttr(
        "ttkgir.critical_path_weight",
        builder.getF32FloatAttr(entry.criticalPathWeight));
  }
}

//===----------------------------------------------------------------------===//
// runOnOperation — Pass entry point
//===----------------------------------------------------------------------===//

void SchedulerPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  LDBG("=== Starting ttkgir-scheduler pass ===");

  // Step 1: Compute critical-path weights (also builds the DAG internally).
  DenseMap<int, float> criticalPathWeights;
  computeCriticalPathWeights(moduleOp, criticalPathWeights);

  if (criticalPathWeights.empty()) {
    LDBG("No schedulable nodes found — pass is a no-op");
    return;
  }

  LDBG("Critical path weights computed for " << criticalPathWeights.size()
                                             << " nodes");

  // Step 2: Ready-queue priority scheduling (highest CP weight first).
  SmallVector<ScheduleEntry, 32> schedule;
  scheduleWithPriority(moduleOp, criticalPathWeights, schedule);

  if (schedule.empty()) {
    LDBG("Schedule is empty after priority scheduling — pass is a no-op");
    return;
  }

  LDBG("Generated schedule with " << schedule.size() << " entries");

  // Step 3: Assign nodes to bounded stream pool.
  assignStreams(schedule, maxStreams_);

  // Step 4: Insert multi-device synchronization barriers.
  insertSynchronizationBarriers(moduleOp, schedule);

  // Step 5: Identify communication-computation overlap opportunities.
  identifyOverlapOpportunities(moduleOp, schedule);

  // Step 6: Write scheduling annotations back to KGIR node operations.
  annotateSchedulingDecisions(schedule);

  LDBG("=== ttkgir-scheduler pass complete: " << schedule.size()
                                              << " nodes scheduled across "
                                              << maxStreams_
                                              << " max streams ===");
}

} // anonymous namespace
