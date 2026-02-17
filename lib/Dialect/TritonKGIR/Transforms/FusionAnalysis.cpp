//===- FusionAnalysis.cpp - KGIR Fusion Analysis Pass ---------------------===//
//
// This file implements the ttkgir-fusion-analysis MLIR pass for the TritonKGIR
// dialect. The pass analyzes a KGIR graph (DAG of kernel launches) for fusion
// opportunities, performing:
//   1. Producer-consumer fusion analysis (kernel pairs sharing an intermediate
//      tensor with single-consumer constraint, tiling compatibility, and
//      combined resource budget per hardware target)
//   2. Sibling/horizontal fusion analysis (independent kernels with compatible
//      grid geometries merged into single launches with partitioned SM/CU
//      allocation)
//   3. Adaptive two-phase cost model evaluation (Phase 1: static heuristic
//      based on eliminated memory traffic and launch overhead; Phase 2:
//      measured runtime data from feedback annotations)
//   4. Per-target fusion plan generation (creates FusedKernelOp nodes and
//      annotates fusion decisions with FusionDecisionAttr)
//
// The pass is part of the graph-level cross-kernel optimization layer.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonKGIR/Transforms/Passes.h"

#include "triton/Dialect/TritonKGIR/IR/Dialect.h"
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
#include <cstdint>

#define DEBUG_TYPE "ttkgir-fusion-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::triton::kgir {
#define GEN_PASS_DEF_TRITONKGIRFUSIONANALYSIS
#include "triton/Dialect/TritonKGIR/Transforms/Passes.h.inc"
} // namespace mlir::triton::kgir

namespace ttkgir = mlir::triton::kgir;

namespace {

// ===== Constants for cost model heuristics ================================ //

// Default shared memory per SM/CU in bytes (48 KB, conservative baseline).
static constexpr int32_t kDefaultSmemPerSmBytes = 49152;

// Default register file size per SM/CU (64K 32-bit registers).
static constexpr int32_t kDefaultRegistersPerSm = 65536;

// Default memory bandwidth in GB/s for cost model normalization.
static constexpr double kDefaultMemoryBandwidthGbps = 900.0;

// Estimated kernel launch overhead in microseconds (GPU-side).
static constexpr double kDefaultLaunchOverheadUs = 5.0;

// Weight for eliminated memory bytes in cost model scoring.
// Represents the inverse throughput penalty of a global memory round-trip.
static constexpr double kBytesEliminationWeight = 1.0e-9;

// Minimum grid dimension product ratio for grid compatibility.
// If the ratio of the smaller to larger grid product exceeds this threshold,
// grids are considered compatible for sibling fusion via partitioning.
static constexpr double kGridCompatibilityRatio = 0.25;

// Maximum number of sibling fusion candidates to consider per graph to
// avoid combinatorial explosion in dense graphs.
static constexpr int32_t kMaxSiblingCandidatesPerGraph = 256;

// Next node ID counter base for fused kernel nodes (starts high to avoid
// collisions with existing kernel node IDs in a graph).
static constexpr int32_t kFusedNodeIdBase = 10000;

// ===== Helper: Encode edge pair for DenseMap key ========================= //

/// Encodes a (producer, consumer) pair into a single int64_t key for use
/// in DenseMap lookups, combining two int32_t node IDs.
static int64_t encodeEdgeKey(int32_t producerId, int32_t consumerId) {
  return (static_cast<int64_t>(static_cast<uint32_t>(producerId)) << 32) |
         static_cast<int64_t>(static_cast<uint32_t>(consumerId));
}

/// Computes the product of grid dimensions as a scalar work-size metric.
static int64_t computeGridProduct(ArrayRef<int64_t> gridDims) {
  int64_t product = 1;
  for (int64_t dim : gridDims) {
    product *= std::max(dim, static_cast<int64_t>(1));
  }
  return product;
}

/// Checks whether two grid dimension arrays are compatible for producer-
/// consumer fusion. Grids are compatible if they have the same number of
/// dimensions and each dimension is either identical or one evenly divides
/// the other.
static bool areGridsCompatibleForPC(ArrayRef<int64_t> producerGrid,
                                    ArrayRef<int64_t> consumerGrid) {
  if (producerGrid.size() != consumerGrid.size())
    return false;
  for (size_t i = 0; i < producerGrid.size(); ++i) {
    int64_t p = std::max(producerGrid[i], static_cast<int64_t>(1));
    int64_t c = std::max(consumerGrid[i], static_cast<int64_t>(1));
    if (p == c)
      continue;
    // Allow divisibility in either direction.
    if (p > c) {
      if (p % c != 0)
        return false;
    } else {
      if (c % p != 0)
        return false;
    }
  }
  return true;
}

/// Checks whether two grid dimension arrays are compatible for sibling
/// fusion. Grids are compatible if they are identical, or their total work
/// sizes (products of all dimensions) are within the compatibility ratio.
static bool areGridsCompatibleForSibling(ArrayRef<int64_t> gridA,
                                         ArrayRef<int64_t> gridB) {
  // Identical grids are always compatible.
  if (gridA.size() == gridB.size()) {
    bool identical = true;
    for (size_t i = 0; i < gridA.size(); ++i) {
      if (gridA[i] != gridB[i]) {
        identical = false;
        break;
      }
    }
    if (identical)
      return true;
  }
  // Check total work-size ratio.
  int64_t productA = computeGridProduct(gridA);
  int64_t productB = computeGridProduct(gridB);
  if (productA == 0 || productB == 0)
    return false;
  double ratio = static_cast<double>(std::min(productA, productB)) /
                 static_cast<double>(std::max(productA, productB));
  return ratio >= kGridCompatibilityRatio;
}

/// Computes the unified grid dimensions for a fused kernel. For producer-
/// consumer fusion, takes the element-wise maximum. For sibling fusion,
/// sums the first dimension for SM partitioning and takes the max of
/// remaining dimensions.
static SmallVector<int64_t, 3>
computeUnifiedGridDims(ArrayRef<int64_t> gridA, ArrayRef<int64_t> gridB,
                       bool isSibling) {
  size_t maxDims = std::max(gridA.size(), gridB.size());
  SmallVector<int64_t, 3> unified(maxDims, 1);
  for (size_t i = 0; i < maxDims; ++i) {
    int64_t a = (i < gridA.size()) ? gridA[i] : 1;
    int64_t b = (i < gridB.size()) ? gridB[i] : 1;
    if (isSibling && i == 0) {
      // For sibling fusion, partition along the first grid dimension by
      // summing block counts to give each kernel its own SM/CU partition.
      unified[i] = a + b;
    } else {
      unified[i] = std::max(a, b);
    }
  }
  return unified;
}

// ===== FusionCandidate: Represents a potential fusion opportunity ========= //

/// Describes a fusion candidate pair with cost model inputs and outputs.
struct FusionCandidate {
  /// Node ID of the producer kernel (first kernel in producer-consumer) or
  /// the first kernel in a sibling pair.
  int32_t producerNodeId = 0;

  /// Node ID of the consumer kernel (second kernel in producer-consumer) or
  /// the second kernel in a sibling pair.
  int32_t consumerNodeId = 0;

  /// Bytes of global memory traffic eliminated by this fusion. For producer-
  /// consumer fusion, this is the intermediate tensor size that no longer
  /// needs to round-trip through global memory.
  int64_t eliminatedBytes = 0;

  /// Estimated speedup ratio from the cost model. Must exceed the pass's
  /// fusionThreshold to be accepted.
  float estimatedSpeedup = 0.0f;

  /// True for sibling/horizontal fusion, false for producer-consumer fusion.
  bool isSibling = false;

  /// Combined shared memory usage in bytes after fusion.
  int32_t combinedSmem = 0;

  /// Combined register pressure after fusion.
  int32_t combinedRegisters = 0;

  /// Unified grid dimensions for the fused kernel.
  SmallVector<int64_t, 3> combinedGridDims;

  /// Pointer to the containing GraphOp for plan generation.
  Operation *parentGraphOp = nullptr;
};

// ===== FusionAnalysis Pass =============================================== //

/// The ttkgir-fusion-analysis pass: Analyzes KGIR graph operations for fusion
/// opportunities and generates per-target fusion plans.
struct FusionAnalysis
    : public ttkgir::impl::TritonKGIRFusionAnalysisBase<FusionAnalysis> {

  // Inherit constructors from the generated base class (picks up
  // fusionThreshold and disableSiblingFusion pass options).
  using TritonKGIRFusionAnalysisBase::TritonKGIRFusionAnalysisBase;

  /// Main entry point: walks all KGIR graphs in the module, identifies fusion
  /// candidates, evaluates the cost model, and generates fusion plans.
  void runOnOperation() override;

private:
  /// Analyzes producer-consumer fusion opportunities within all GraphOps.
  /// Identifies kernel pairs where Kernel A writes a tensor that Kernel B
  /// reads, with no other consumers, compatible tiling, and combined resources
  /// within per-SM/CU limits.
  void analyzeProducerConsumerFusion(
      ModuleOp moduleOp, SmallVectorImpl<FusionCandidate> &candidates);

  /// Analyzes sibling/horizontal fusion opportunities within all GraphOps.
  /// Identifies independent kernel pairs with compatible grid geometries that
  /// can be merged into single launches with partitioned SM/CU allocation.
  void analyzeSiblingFusion(ModuleOp moduleOp,
                            SmallVectorImpl<FusionCandidate> &candidates);

  /// Evaluates the adaptive two-phase cost model for a fusion candidate.
  /// Phase 1 (cold start): heuristic based on eliminated bytes, launch
  /// overhead savings, and resource pressure penalty.
  /// Phase 2 (after feedback): uses measured runtime performance data when
  /// RuntimePerformanceAttr annotations are present on KGIR nodes.
  /// Returns the estimated speedup ratio.
  float evaluateCostModel(const FusionCandidate &candidate,
                          const llvm::DenseMap<int32_t, Operation *> &nodeMap);

  /// Checks fusion legality for a producer-consumer pair. Validates:
  ///   - Single-consumer constraint (no other kernel reads the intermediate)
  ///   - Tiling compatibility (grid dimensions are compatible)
  ///   - Combined SMEM + register budget within per-target hardware limits
  /// Returns true if fusion is legal, and populates output parameters.
  bool isProducerConsumerLegal(
      Operation *producer, Operation *consumer,
      int64_t &eliminatedBytes, int32_t &combinedSmem,
      int32_t &combinedRegisters, SmallVectorImpl<int64_t> &combinedGridDims,
      const llvm::DenseMap<int32_t, SmallVector<int32_t>> &producerToConsumers);

  /// Checks fusion legality for a sibling pair. Validates:
  ///   - Independence (no data dependency between the pair)
  ///   - Grid geometry compatibility
  ///   - Combined resource limits (SMEM + registers)
  /// Returns true if fusion is legal, and populates output parameters.
  bool isSiblingFusionLegal(Operation *kernelA, Operation *kernelB,
                            int32_t &combinedSmem,
                            int32_t &combinedRegisters,
                            SmallVectorImpl<int64_t> &combinedGridDims);

  /// Generates the per-target fusion plan from accepted candidates. For each
  /// accepted fusion, creates a ttkgir.fused_kernel operation inside the
  /// containing GraphOp and annotates it with FusionDecisionAttr. Also marks
  /// original KernelLaunchOps with discardable fusion decision attributes.
  void generateFusionPlan(
      ModuleOp moduleOp,
      const SmallVectorImpl<FusionCandidate> &candidates);
};

} // namespace

// ===== runOnOperation Implementation ===================================== //

void FusionAnalysis::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  LDBG("Starting fusion analysis on KGIR graph");
  LDBG("Fusion threshold: " << fusionThreshold);
  LDBG("Sibling fusion: " << (disableSiblingFusion ? "disabled" : "enabled"));

  SmallVector<FusionCandidate> candidates;

  // Step 1: Identify producer-consumer fusion opportunities.
  analyzeProducerConsumerFusion(moduleOp, candidates);
  size_t pcCount = candidates.size();
  LDBG("Found " << pcCount << " producer-consumer candidates");

  // Step 2: Identify sibling/horizontal fusion opportunities (unless disabled
  // via the --disable-sibling-fusion pass option).
  if (!disableSiblingFusion) {
    analyzeSiblingFusion(moduleOp, candidates);
    LDBG("Found " << (candidates.size() - pcCount) << " sibling candidates");
  }

  LDBG("Total fusion candidates: " << candidates.size());

  if (candidates.empty()) {
    LDBG("No fusion candidates found — pass is a no-op");
    return;
  }

  // Step 3: Build node-ID → Operation* map for cost model evaluation.
  llvm::DenseMap<int32_t, Operation *> nodeMap;
  moduleOp.walk([&](ttkgir::KernelLaunchOp launchOp) {
    int32_t nodeId =
        static_cast<int32_t>(launchOp.getNodeIdAttr().getInt());
    nodeMap[nodeId] = launchOp.getOperation();
  });

  // Step 4: Evaluate the cost model for each candidate and filter by the
  // fusion threshold.
  SmallVector<FusionCandidate> acceptedCandidates;
  for (auto &candidate : candidates) {
    float speedup = evaluateCostModel(candidate, nodeMap);
    candidate.estimatedSpeedup = speedup;
    if (speedup >= fusionThreshold) {
      acceptedCandidates.push_back(candidate);
      LDBG("Accepted "
           << (candidate.isSibling ? "sibling" : "producer-consumer")
           << " fusion: node " << candidate.producerNodeId << " + node "
           << candidate.consumerNodeId << " (speedup: " << speedup << ")");
    } else {
      LDBG("Rejected "
           << (candidate.isSibling ? "sibling" : "producer-consumer")
           << " fusion: node " << candidate.producerNodeId << " + node "
           << candidate.consumerNodeId << " (speedup: " << speedup
           << " < threshold " << fusionThreshold << ")");
    }
  }

  LDBG("Accepted " << acceptedCandidates.size() << " of " << candidates.size()
                    << " candidates");

  // Step 5: Generate per-target fusion plan for accepted candidates.
  if (!acceptedCandidates.empty()) {
    generateFusionPlan(moduleOp, acceptedCandidates);
    LDBG("Fusion plan generated with " << acceptedCandidates.size()
                                        << " fused kernel(s)");
  }
}

// ===== analyzeProducerConsumerFusion Implementation ====================== //

void FusionAnalysis::analyzeProducerConsumerFusion(
    ModuleOp moduleOp, SmallVectorImpl<FusionCandidate> &candidates) {

  moduleOp.walk([&](ttkgir::GraphOp graphOp) {
    LDBG("Analyzing producer-consumer fusion in graph: "
         << graphOp.getGraphName());

    // Collect all KernelLaunchOps by node ID within this graph.
    llvm::DenseMap<int32_t, Operation *> kernelsByNodeId;
    graphOp.getBody().walk([&](ttkgir::KernelLaunchOp launchOp) {
      int32_t nodeId =
          static_cast<int32_t>(launchOp.getNodeIdAttr().getInt());
      kernelsByNodeId[nodeId] = launchOp.getOperation();
    });

    if (kernelsByNodeId.empty()) {
      LDBG("  No kernel launches in graph — skipping");
      return WalkResult::advance();
    }

    // Build adjacency map from DataDepOps: producer → list of consumers.
    // Only consider "flow" dependencies (read-after-write) since those
    // represent true data dependencies suitable for producer-consumer fusion.
    llvm::DenseMap<int32_t, SmallVector<int32_t>> producerToConsumers;
    // Track eliminated tensor sizes from data dependency edges.
    llvm::DenseMap<int64_t, int64_t> edgeTensorSize;

    graphOp.getBody().walk([&](ttkgir::DataDepOp depOp) {
      StringRef depType = depOp.getDepType();
      if (depType != "flow")
        return WalkResult::advance();

      int32_t srcId =
          static_cast<int32_t>(depOp.getSourceNodeIdAttr().getInt());
      int32_t dstId =
          static_cast<int32_t>(depOp.getDestNodeIdAttr().getInt());

      producerToConsumers[srcId].push_back(dstId);

      // Estimate the intermediate tensor size from the producer's memory
      // access pattern if available, for cost model evaluation.
      int64_t tensorBytes = 0;
      auto producerIt = kernelsByNodeId.find(srcId);
      if (producerIt != kernelsByNodeId.end()) {
        auto producerOp =
            dyn_cast<ttkgir::KernelLaunchOp>(producerIt->second);
        if (producerOp) {
          auto memPat = producerOp.getMemoryAccessPatternsAttr();
          if (memPat) {
            tensorBytes = memPat.getAccessSizeBytes();
          }
        }
      }
      // Store with edge key for cost model lookup.
      edgeTensorSize[encodeEdgeKey(srcId, dstId)] = tensorBytes;

      return WalkResult::advance();
    });

    LDBG("  Built adjacency map: " << producerToConsumers.size()
                                    << " producers with flow deps");

    // For each producer with exactly one consumer (single-consumer constraint),
    // check fusion legality.
    for (auto &[producerId, consumers] : producerToConsumers) {
      if (consumers.size() != 1)
        continue;  // Multi-consumer — not eligible for P-C fusion.

      int32_t consumerId = consumers[0];

      auto producerIt = kernelsByNodeId.find(producerId);
      auto consumerIt = kernelsByNodeId.find(consumerId);
      if (producerIt == kernelsByNodeId.end() ||
          consumerIt == kernelsByNodeId.end()) {
        LDBG("  Missing kernel for edge " << producerId << " -> "
                                           << consumerId);
        continue;
      }

      int64_t eliminatedBytes = 0;
      int32_t combinedSmem = 0;
      int32_t combinedRegisters = 0;
      SmallVector<int64_t, 3> combinedGridDims;

      if (isProducerConsumerLegal(producerIt->second, consumerIt->second,
                                  eliminatedBytes, combinedSmem,
                                  combinedRegisters, combinedGridDims,
                                  producerToConsumers)) {
        // Override eliminated bytes with edge-specific data if available.
        int64_t edgeKey = encodeEdgeKey(producerId, consumerId);
        auto edgeIt = edgeTensorSize.find(edgeKey);
        if (edgeIt != edgeTensorSize.end() && edgeIt->second > 0) {
          eliminatedBytes = edgeIt->second;
        }

        FusionCandidate candidate;
        candidate.producerNodeId = producerId;
        candidate.consumerNodeId = consumerId;
        candidate.eliminatedBytes = eliminatedBytes;
        candidate.estimatedSpeedup = 0.0f;  // Computed later by cost model.
        candidate.isSibling = false;
        candidate.combinedSmem = combinedSmem;
        candidate.combinedRegisters = combinedRegisters;
        candidate.combinedGridDims = std::move(combinedGridDims);
        candidate.parentGraphOp = graphOp.getOperation();
        candidates.push_back(std::move(candidate));

        LDBG("  P-C candidate: " << producerId << " -> " << consumerId
                                  << " (elim=" << eliminatedBytes << " B)");
      }
    }

    return WalkResult::advance();
  });
}

// ===== analyzeSiblingFusion Implementation =============================== //

void FusionAnalysis::analyzeSiblingFusion(
    ModuleOp moduleOp, SmallVectorImpl<FusionCandidate> &candidates) {

  moduleOp.walk([&](ttkgir::GraphOp graphOp) {
    LDBG("Analyzing sibling fusion in graph: " << graphOp.getGraphName());

    // Collect all KernelLaunchOps.
    SmallVector<std::pair<int32_t, Operation *>> kernels;
    graphOp.getBody().walk([&](ttkgir::KernelLaunchOp launchOp) {
      int32_t nodeId =
          static_cast<int32_t>(launchOp.getNodeIdAttr().getInt());
      kernels.push_back({nodeId, launchOp.getOperation()});
      return WalkResult::advance();
    });

    if (kernels.size() < 2) {
      LDBG("  Fewer than 2 kernels — no sibling pairs possible");
      return WalkResult::advance();
    }

    // Build a set of related pairs (connected by any type of dependency).
    llvm::DenseSet<int64_t> dependentPairs;

    // Collect data dependencies.
    graphOp.getBody().walk([&](ttkgir::DataDepOp depOp) {
      int32_t srcId =
          static_cast<int32_t>(depOp.getSourceNodeIdAttr().getInt());
      int32_t dstId =
          static_cast<int32_t>(depOp.getDestNodeIdAttr().getInt());
      dependentPairs.insert(encodeEdgeKey(srcId, dstId));
      dependentPairs.insert(encodeEdgeKey(dstId, srcId));
      return WalkResult::advance();
    });

    // Collect anti-dependencies.
    graphOp.getBody().walk([&](ttkgir::AntiDepOp antiOp) {
      int32_t srcId =
          static_cast<int32_t>(antiOp.getSourceNodeIdAttr().getInt());
      int32_t dstId =
          static_cast<int32_t>(antiOp.getDestNodeIdAttr().getInt());
      dependentPairs.insert(encodeEdgeKey(srcId, dstId));
      dependentPairs.insert(encodeEdgeKey(dstId, srcId));
      return WalkResult::advance();
    });

    // Enumerate independent kernel pairs and check sibling fusion legality.
    int32_t siblingCount = 0;
    for (size_t i = 0; i < kernels.size() && siblingCount < kMaxSiblingCandidatesPerGraph; ++i) {
      for (size_t j = i + 1; j < kernels.size() && siblingCount < kMaxSiblingCandidatesPerGraph; ++j) {
        int32_t idA = kernels[i].first;
        int32_t idB = kernels[j].first;

        // Skip if there is any dependency between the pair.
        if (dependentPairs.count(encodeEdgeKey(idA, idB)))
          continue;

        int32_t combinedSmem = 0;
        int32_t combinedRegisters = 0;
        SmallVector<int64_t, 3> combinedGridDims;

        if (isSiblingFusionLegal(kernels[i].second, kernels[j].second,
                                 combinedSmem, combinedRegisters,
                                 combinedGridDims)) {
          FusionCandidate candidate;
          candidate.producerNodeId = idA;
          candidate.consumerNodeId = idB;
          candidate.eliminatedBytes = 0;  // Siblings share no intermediate.
          candidate.estimatedSpeedup = 0.0f;
          candidate.isSibling = true;
          candidate.combinedSmem = combinedSmem;
          candidate.combinedRegisters = combinedRegisters;
          candidate.combinedGridDims = std::move(combinedGridDims);
          candidate.parentGraphOp = graphOp.getOperation();
          candidates.push_back(std::move(candidate));
          ++siblingCount;

          LDBG("  Sibling candidate: " << idA << " + " << idB);
        }
      }
    }

    LDBG("  Found " << siblingCount << " sibling candidates in graph");
    return WalkResult::advance();
  });
}

// ===== evaluateCostModel Implementation ================================== //

float FusionAnalysis::evaluateCostModel(
    const FusionCandidate &candidate,
    const llvm::DenseMap<int32_t, Operation *> &nodeMap) {

  // ---- Phase 2: Use measured runtime data if available ---- //
  // Check if both kernels have RuntimePerformanceAttr annotations (written
  // back by the feedback controller after profiled execution).
  auto producerIt = nodeMap.find(candidate.producerNodeId);
  auto consumerIt = nodeMap.find(candidate.consumerNodeId);
  bool hasRuntimeData = false;
  double producerWallClockUs = 0.0;
  double consumerWallClockUs = 0.0;
  double producerLaunchOverheadUs = kDefaultLaunchOverheadUs;
  double consumerLaunchOverheadUs = kDefaultLaunchOverheadUs;

  if (producerIt != nodeMap.end() && consumerIt != nodeMap.end()) {
    auto producerOp = dyn_cast<ttkgir::KernelLaunchOp>(producerIt->second);
    auto consumerOp = dyn_cast<ttkgir::KernelLaunchOp>(consumerIt->second);

    if (producerOp && consumerOp) {
      auto producerPerf = producerOp.getRuntimePerfAttr();
      auto consumerPerf = consumerOp.getRuntimePerfAttr();

      if (producerPerf && consumerPerf) {
        hasRuntimeData = true;
        producerWallClockUs = producerPerf.getWallClockUs();
        consumerWallClockUs = consumerPerf.getWallClockUs();
        producerLaunchOverheadUs = producerPerf.getLaunchOverheadUs();
        consumerLaunchOverheadUs = consumerPerf.getLaunchOverheadUs();
        LDBG("  Phase 2 cost model: using measured data for nodes "
             << candidate.producerNodeId << " + "
             << candidate.consumerNodeId);
      }
    }
  }

  if (hasRuntimeData) {
    // Phase 2 cost model: use measured execution times.
    // Estimated time without fusion = sum of both wall-clock times.
    double unfusedTimeUs =
        producerWallClockUs + consumerWallClockUs;

    // Fused kernel eliminates one launch overhead and the global memory
    // round-trip for the intermediate tensor.
    double eliminatedLaunchUs = consumerLaunchOverheadUs;

    // Memory bandwidth savings: intermediate tensor bytes / measured
    // throughput. Use producer's throughput if available.
    double memorySavingsUs = 0.0;
    if (candidate.eliminatedBytes > 0) {
      auto prodOp = dyn_cast<ttkgir::KernelLaunchOp>(producerIt->second);
      if (prodOp) {
        auto perf = prodOp.getRuntimePerfAttr();
        if (perf && perf.getMemoryThroughputGbps() > 0.0) {
          double throughputBytesPerUs =
              perf.getMemoryThroughputGbps() * 1e3;  // GB/s → MB/us → B/us
          memorySavingsUs =
              static_cast<double>(candidate.eliminatedBytes) /
              throughputBytesPerUs;
        }
      }
    }

    // Resource pressure penalty based on combined SMEM usage.
    double resourcePenalty =
        1.0 + static_cast<double>(candidate.combinedSmem) /
                  static_cast<double>(kDefaultSmemPerSmBytes);

    double fusedTimeUs =
        (unfusedTimeUs - eliminatedLaunchUs - memorySavingsUs) *
        resourcePenalty;

    // Clamp fused time to be positive.
    fusedTimeUs = std::max(fusedTimeUs, 0.1);

    double speedup = unfusedTimeUs / fusedTimeUs - 1.0;
    return static_cast<float>(std::max(speedup, 0.0));
  }

  // ---- Phase 1: Static heuristic cost model ---- //
  // Component 1: Eliminated memory traffic benefit.
  // Score for removing the global memory write-then-read of the intermediate
  // tensor. Normalized by memory bandwidth to produce a time-equivalent.
  double eliminatedBytesScore =
      static_cast<double>(candidate.eliminatedBytes) * kBytesEliminationWeight;

  // Component 2: Launch overhead savings.
  // Fusing two kernels saves one GPU kernel launch worth of overhead.
  // Normalized to a fraction relative to typical kernel execution time.
  double launchSavingsScore = kDefaultLaunchOverheadUs / 100.0;

  // For sibling fusion, the primary benefit is reducing total launch count.
  // There's no intermediate tensor elimination.
  if (candidate.isSibling) {
    eliminatedBytesScore = 0.0;
    // Sibling fusion also enables potential SM partitioning benefits —
    // model as a fraction of launch overhead savings.
    launchSavingsScore *= 1.5;
  }

  // Component 3: Resource pressure penalty.
  // Higher combined resource usage reduces occupancy and may negate gains.
  double smemUtilization =
      static_cast<double>(candidate.combinedSmem) /
      static_cast<double>(kDefaultSmemPerSmBytes);
  double regUtilization =
      static_cast<double>(candidate.combinedRegisters) /
      static_cast<double>(kDefaultRegistersPerSm);
  double resourcePenalty =
      1.0 + 0.5 * smemUtilization + 0.3 * regUtilization;

  // Final heuristic speedup estimate.
  double rawBenefit = eliminatedBytesScore + launchSavingsScore;
  double speedup = rawBenefit / resourcePenalty;

  LDBG("  Phase 1 cost: elim=" << eliminatedBytesScore
                                << " launch=" << launchSavingsScore
                                << " penalty=" << resourcePenalty
                                << " speedup=" << speedup);

  return static_cast<float>(std::max(speedup, 0.0));
}

// ===== isProducerConsumerLegal Implementation ============================ //

bool FusionAnalysis::isProducerConsumerLegal(
    Operation *producer, Operation *consumer,
    int64_t &eliminatedBytes, int32_t &combinedSmem,
    int32_t &combinedRegisters, SmallVectorImpl<int64_t> &combinedGridDims,
    const llvm::DenseMap<int32_t, SmallVector<int32_t>> &producerToConsumers) {

  auto producerOp = dyn_cast<ttkgir::KernelLaunchOp>(producer);
  auto consumerOp = dyn_cast<ttkgir::KernelLaunchOp>(consumer);

  if (!producerOp || !consumerOp) {
    LDBG("    Legality: non-KernelLaunchOp operand — illegal");
    return false;
  }

  int32_t producerId =
      static_cast<int32_t>(producerOp.getNodeIdAttr().getInt());

  // ---- Check 1: Single-consumer constraint ---- //
  // The producer must have exactly one consumer for the intermediate tensor
  // to be fully eliminated. Multi-consumer intermediates cannot be promoted
  // to shared memory without potential data duplication.
  auto it = producerToConsumers.find(producerId);
  if (it == producerToConsumers.end() || it->second.size() != 1) {
    LDBG("    Legality: producer " << producerId
                                    << " has multiple consumers — illegal");
    return false;
  }

  // ---- Check 2: Tiling / grid compatibility ---- //
  ArrayRef<int64_t> producerGrid = producerOp.getGridDims();
  ArrayRef<int64_t> consumerGrid = consumerOp.getGridDims();

  if (!areGridsCompatibleForPC(producerGrid, consumerGrid)) {
    LDBG("    Legality: incompatible grid dims — illegal");
    return false;
  }

  // ---- Check 3: Combined resource budget ---- //
  int32_t producerSmem =
      static_cast<int32_t>(producerOp.getSharedMemoryBytesAttr().getInt());
  int32_t consumerSmem =
      static_cast<int32_t>(consumerOp.getSharedMemoryBytesAttr().getInt());
  combinedSmem = producerSmem + consumerSmem;

  int32_t producerRegs =
      static_cast<int32_t>(producerOp.getRegisterPressureAttr().getInt());
  int32_t consumerRegs =
      static_cast<int32_t>(consumerOp.getRegisterPressureAttr().getInt());
  combinedRegisters = producerRegs + consumerRegs;

  // Check against default hardware limits. When HardwareProfileType metadata
  // is available on the containing GraphOp, per-target limits would be used
  // instead. For the static analysis pass, use conservative defaults.
  int32_t maxSmem = kDefaultSmemPerSmBytes;
  int32_t maxRegs = kDefaultRegistersPerSm;

  // Try to extract per-target limits from the producer's hardware target
  // annotation if available. This provides better accuracy when hardware
  // profile information has been populated during trace capture.
  auto hwTarget = producerOp.getHardwareTargetAttr();
  (void)hwTarget;  // Hardware target annotation is informational at this stage;
                    // per-target limit lookup would require resolving the full
                    // HardwareProfile from the GraphOp's hardware_profiles JSON.
                    // For static analysis, defaults are sufficient.

  if (combinedSmem > maxSmem) {
    LDBG("    Legality: combined SMEM " << combinedSmem << " > limit "
                                         << maxSmem << " — illegal");
    return false;
  }

  if (combinedRegisters > maxRegs) {
    LDBG("    Legality: combined registers " << combinedRegisters
                                              << " > limit " << maxRegs
                                              << " — illegal");
    return false;
  }

  // ---- Compute combined grid dimensions ---- //
  auto unified = computeUnifiedGridDims(producerGrid, consumerGrid,
                                        /*isSibling=*/false);
  combinedGridDims.assign(unified.begin(), unified.end());

  // ---- Estimate eliminated bytes ---- //
  // Use the producer's memory access pattern to estimate the size of the
  // intermediate tensor that would be eliminated by fusion.
  eliminatedBytes = 0;
  auto memPat = producerOp.getMemoryAccessPatternsAttr();
  if (memPat) {
    eliminatedBytes = memPat.getAccessSizeBytes();
  } else {
    // Heuristic fallback: estimate based on grid dimensions and a default
    // element size (4 bytes per float). This is a rough approximation.
    int64_t gridProduct = computeGridProduct(producerGrid);
    eliminatedBytes = gridProduct * 4;  // Assume fp32 intermediate.
  }

  LDBG("    Legality: LEGAL (smem=" << combinedSmem
                                     << ", regs=" << combinedRegisters
                                     << ", elim=" << eliminatedBytes << " B)");
  return true;
}

// ===== isSiblingFusionLegal Implementation =============================== //

bool FusionAnalysis::isSiblingFusionLegal(
    Operation *kernelA, Operation *kernelB,
    int32_t &combinedSmem, int32_t &combinedRegisters,
    SmallVectorImpl<int64_t> &combinedGridDims) {

  auto opA = dyn_cast<ttkgir::KernelLaunchOp>(kernelA);
  auto opB = dyn_cast<ttkgir::KernelLaunchOp>(kernelB);

  if (!opA || !opB) {
    LDBG("    Sibling legality: non-KernelLaunchOp — illegal");
    return false;
  }

  // ---- Check 1: Grid geometry compatibility ---- //
  ArrayRef<int64_t> gridA = opA.getGridDims();
  ArrayRef<int64_t> gridB = opB.getGridDims();

  if (!areGridsCompatibleForSibling(gridA, gridB)) {
    LDBG("    Sibling legality: incompatible grids — illegal");
    return false;
  }

  // ---- Check 2: Combined resource limits ---- //
  int32_t smemA =
      static_cast<int32_t>(opA.getSharedMemoryBytesAttr().getInt());
  int32_t smemB =
      static_cast<int32_t>(opB.getSharedMemoryBytesAttr().getInt());
  combinedSmem = smemA + smemB;

  int32_t regsA =
      static_cast<int32_t>(opA.getRegisterPressureAttr().getInt());
  int32_t regsB =
      static_cast<int32_t>(opB.getRegisterPressureAttr().getInt());
  combinedRegisters = regsA + regsB;

  // Check against conservative hardware limits.
  if (combinedSmem > kDefaultSmemPerSmBytes) {
    LDBG("    Sibling legality: combined SMEM " << combinedSmem << " exceeds "
                                                 << kDefaultSmemPerSmBytes);
    return false;
  }
  if (combinedRegisters > kDefaultRegistersPerSm) {
    LDBG("    Sibling legality: combined registers " << combinedRegisters
                                                      << " exceeds "
                                                      << kDefaultRegistersPerSm);
    return false;
  }

  // ---- Compute unified grid dimensions (sibling: SM partitioning) ---- //
  auto unified =
      computeUnifiedGridDims(gridA, gridB, /*isSibling=*/true);
  combinedGridDims.assign(unified.begin(), unified.end());

  LDBG("    Sibling legality: LEGAL (smem=" << combinedSmem
                                             << ", regs=" << combinedRegisters
                                             << ")");
  return true;
}

// ===== generateFusionPlan Implementation ================================= //

void FusionAnalysis::generateFusionPlan(
    ModuleOp moduleOp,
    const SmallVectorImpl<FusionCandidate> &candidates) {

  MLIRContext *ctx = moduleOp.getContext();

  // Group candidates by their parent GraphOp for efficient insertion.
  llvm::DenseMap<Operation *, SmallVector<const FusionCandidate *>>
      candidatesByGraph;
  for (const auto &candidate : candidates) {
    if (candidate.parentGraphOp) {
      candidatesByGraph[candidate.parentGraphOp].push_back(&candidate);
    }
  }

  // Track which nodes have been assigned to a fusion to prevent overlapping
  // fusions (a node should participate in at most one fusion).
  llvm::DenseSet<int32_t> fusedNodes;

  // Counter for generating unique fused node IDs.
  int32_t nextFusedNodeId = kFusedNodeIdBase;

  for (auto &[graphOperation, graphCandidates] : candidatesByGraph) {
    auto graphOp = dyn_cast<ttkgir::GraphOp>(graphOperation);
    if (!graphOp)
      continue;

    // Build node-ID → KernelLaunchOp map within this graph for annotation.
    llvm::DenseMap<int32_t, ttkgir::KernelLaunchOp> launchOps;
    graphOp.getBody().walk([&](ttkgir::KernelLaunchOp launchOp) {
      int32_t nodeId =
          static_cast<int32_t>(launchOp.getNodeIdAttr().getInt());
      launchOps[nodeId] = launchOp;
      return WalkResult::advance();
    });

    // Create an OpBuilder positioned at the end of the graph's body block.
    Block &block = graphOp.getBody().front();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(&block);
    Location loc = graphOp.getLoc();

    for (const FusionCandidate *candidate : graphCandidates) {
      // Prevent overlapping fusions: each node participates in at most one.
      if (fusedNodes.count(candidate->producerNodeId) ||
          fusedNodes.count(candidate->consumerNodeId)) {
        LDBG("  Skipping overlapping fusion: nodes "
             << candidate->producerNodeId << " and "
             << candidate->consumerNodeId << " already fused");
        continue;
      }

      // Determine fusion type string.
      StringRef fusionTypeStr =
          candidate->isSibling ? "sibling" : "producer_consumer";

      // Build the fused kernel name from the constituent kernel names.
      std::string fusedName;
      auto producerLaunchIt = launchOps.find(candidate->producerNodeId);
      auto consumerLaunchIt = launchOps.find(candidate->consumerNodeId);
      if (producerLaunchIt != launchOps.end() &&
          consumerLaunchIt != launchOps.end()) {
        StringRef prodName = producerLaunchIt->second.getKernelName();
        StringRef consName = consumerLaunchIt->second.getKernelName();
        fusedName =
            ("fused_" + prodName + "_" + consName).str();
      } else {
        fusedName = "fused_" + std::to_string(candidate->producerNodeId) +
                    "_" + std::to_string(candidate->consumerNodeId);
      }

      // Build the fused node IDs array.
      SmallVector<int32_t, 2> fusedNodeIds = {candidate->producerNodeId,
                                               candidate->consumerNodeId};

      // Determine target device ID from the producer's hardware target
      // annotation, defaulting to 0 if not available.
      int32_t targetDeviceId = 0;
      if (producerLaunchIt != launchOps.end()) {
        auto hwTarget = producerLaunchIt->second.getHardwareTargetAttr();
        if (hwTarget) {
          targetDeviceId =
              static_cast<int32_t>(hwTarget.getDeviceId());
        }
      }

      // Create FusionDecisionAttr to annotate the fused kernel with the
      // analysis result.
      std::string reason;
      if (candidate->isSibling) {
        reason = "sibling_fusion_grid_compatible";
      } else {
        reason = "pc_fusion_single_consumer_elim_" +
                 std::to_string(candidate->eliminatedBytes) + "B";
      }

      auto fusionDecision = ttkgir::FusionDecisionAttr::get(
          ctx,
          /*is_fused=*/true,
          /*fusion_type=*/fusionTypeStr,
          /*reason=*/reason,
          /*estimated_speedup=*/static_cast<double>(candidate->estimatedSpeedup),
          /*target_device_id=*/static_cast<int64_t>(targetDeviceId));

      // Assign a unique node ID for the fused kernel.
      int32_t fusedNodeId = nextFusedNodeId++;

      // Create the FusedKernelOp using OperationState for robustness against
      // TableGen-generated create() signature variations.
      OperationState state(loc, ttkgir::FusedKernelOp::getOperationName());
      state.addAttribute("fused_name", builder.getStringAttr(fusedName));
      state.addAttribute("fused_node_ids",
                         builder.getDenseI32ArrayAttr(fusedNodeIds));
      state.addAttribute("fusion_type",
                         builder.getStringAttr(fusionTypeStr));
      state.addAttribute("combined_grid_dims",
                         builder.getDenseI64ArrayAttr(
                             candidate->combinedGridDims));
      state.addAttribute("combined_shared_memory_bytes",
                         builder.getI32IntegerAttr(candidate->combinedSmem));
      state.addAttribute("combined_register_pressure",
                         builder.getI32IntegerAttr(
                             candidate->combinedRegisters));
      state.addAttribute("target_device_id",
                         builder.getI32IntegerAttr(targetDeviceId));
      state.addAttribute("fusion_decision", fusionDecision);
      state.addAttribute("node_id",
                         builder.getI32IntegerAttr(fusedNodeId));

      builder.create(state);

      LDBG("  Created FusedKernelOp '" << fusedName << "' (node "
                                        << fusedNodeId << ") for "
                                        << fusionTypeStr << " fusion of nodes "
                                        << candidate->producerNodeId << " + "
                                        << candidate->consumerNodeId);

      // Annotate the original KernelLaunchOps with discardable fusion
      // decision attributes so downstream passes can identify fused kernels
      // without re-running the analysis.
      auto annotateKernel = [&](int32_t nodeId, StringRef role) {
        auto it = launchOps.find(nodeId);
        if (it != launchOps.end()) {
          auto annotation = ttkgir::FusionDecisionAttr::get(
              ctx,
              /*is_fused=*/true,
              /*fusion_type=*/fusionTypeStr,
              /*reason=*/(role + "_in_" + fusedName).str(),
              /*estimated_speedup=*/
              static_cast<double>(candidate->estimatedSpeedup),
              /*target_device_id=*/static_cast<int64_t>(targetDeviceId));
          it->second->setAttr("ttkgir.fusion_decision", annotation);
        }
      };

      annotateKernel(candidate->producerNodeId, "producer");
      annotateKernel(candidate->consumerNodeId, "consumer");

      // Mark both nodes as fused to prevent overlapping fusions.
      fusedNodes.insert(candidate->producerNodeId);
      fusedNodes.insert(candidate->consumerNodeId);
    }
  }
}
