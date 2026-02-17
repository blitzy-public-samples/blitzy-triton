#ifndef TRITONKGIR_PASSES
#define TRITONKGIR_PASSES

include "mlir/Pass/PassBase.td"

def TritonKGIRFusionAnalysis : Pass<"ttkgir-fusion-analysis", "mlir::ModuleOp"> {
  let summary = "Analyze KGIR graph for fusion opportunities";

  let description = [{
    The `ttkgir-fusion-analysis` pass analyzes the KGIR graph for producer-consumer
    and sibling fusion opportunities. It performs fusion legality checking, cost model
    evaluation, and generates per-target fusion plans. Producer-consumer fusion
    identifies pairs where Kernel A writes a tensor that Kernel B reads with no
    other consumers, compatible tiling, and combined resources within per-SM/CU
    limits. Sibling fusion identifies independent kernels with compatible grid
    geometries that can be merged into single launches with partitioned SM/CU
    allocation.
  }];

  let dependentDialects = [
    "mlir::triton::kgir::TritonKGIRDialect",
  ];

  let options = [
    Option<"fusionThreshold", "fusion-threshold",
           "float", /*default*/"0.10",
           "Minimum estimated speedup threshold for fusion decisions">,
    Option<"disableSiblingFusion", "disable-sibling-fusion",
           "bool", /*default*/"false",
           "Disable sibling/horizontal fusion analysis">
  ];
}

def TritonKGIRMemoryPlanning : Pass<"ttkgir-memory-planning", "mlir::ModuleOp"> {
  let summary = "Plan memory allocation across KGIR graph kernels";

  let description = [{
    The `ttkgir-memory-planning` pass performs liveness analysis on the KGIR graph,
    identifies intermediate tensors produced and consumed within the graph, and
    promotes eligible intermediates from global memory to shared memory or register
    file across fused kernels. Promotion decisions respect per-target hardware limits
    from HardwareProfile descriptors. The pass also inserts cross-device transfer
    operations when dispatch splits graphs across devices.
  }];

  let dependentDialects = [
    "mlir::triton::kgir::TritonKGIRDialect",
  ];
}

def TritonKGIRScheduler : Pass<"ttkgir-scheduler", "mlir::ModuleOp"> {
  let summary = "Schedule KGIR graph kernels for concurrent execution";

  let description = [{
    The `ttkgir-scheduler` pass performs critical-path analysis on the KGIR DAG,
    assigns kernels to streams for concurrent execution, applies resource-aware
    SM/CU bin-packing, inserts multi-device synchronization barriers at cross-device
    edges, and identifies communication-computation overlap opportunities.
  }];

  let dependentDialects = [
    "mlir::triton::kgir::TritonKGIRDialect",
  ];
}

#endif
