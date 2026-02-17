//===-- TritonKGIRAttrDefs.td - KGIR Attribute Defs --------*- tablegen -*-===//
//
// Part of the Triton project.
//
// Defines attribute types for the TritonKGIR (Kernel Graph IR) dialect.
// These attributes annotate KGIR nodes and edges with memory access patterns,
// hardware target metadata, runtime performance measurements, and fusion
// decision information used by the graph-level cross-kernel optimization layer.
//
//===----------------------------------------------------------------------===//

#ifndef TRITONKGIR_ATTRDEFS
#define TRITONKGIR_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonKGIR/IR/TritonKGIRDialect.td"

//===----------------------------------------------------------------------===//
// MemoryAccessPatternAttr
//===----------------------------------------------------------------------===//

def TTKGIR_MemoryAccessPatternAttr
    : AttrDef<TritonKGIR_Dialect, "MemoryAccessPattern"> {
  let mnemonic = "mem_access_pattern";
  let attrName = "ttkgir.mem_access_pattern";

  let description = [{
    Describes the memory access pattern for a single tensor argument of a
    kernel node in the KGIR graph.

    Each kernel node may carry zero or more MemoryAccessPattern attributes,
    one per tensor argument, capturing whether the argument is read, written,
    or both, whether the access is contiguous in memory, and the total number
    of bytes accessed. This information drives fusion analysis (identifying
    producer-consumer pairs where one kernel writes a tensor that another
    reads) and memory planning (determining which intermediates can be
    promoted from global memory to shared memory or registers).

    The `pattern_type` field accepts the values "read", "write", or
    "readwrite" indicating the direction of the memory access.
  }];

  let parameters = (ins
    StringRefParameter<"access pattern type: read, write, or readwrite">:$pattern_type,
    "int64_t":$tensor_index,
    DefaultValuedParameter<"bool", "true",
        "whether the memory access is contiguous">:$is_contiguous,
    "int64_t":$access_size_bytes
  );

  let assemblyFormat = "`<` $pattern_type `,` $tensor_index `,` $is_contiguous `,` $access_size_bytes `>`";
}

//===----------------------------------------------------------------------===//
// HardwareTargetAnnotationAttr
//===----------------------------------------------------------------------===//

def TTKGIR_HardwareTargetAnnotationAttr
    : AttrDef<TritonKGIR_Dialect, "HardwareTargetAnnotation"> {
  let mnemonic = "hw_target";
  let attrName = "ttkgir.hw_target";

  let description = [{
    Hardware target metadata attached to KGIR nodes for dispatch decisions.

    Each kernel node in the KGIR graph may be annotated with a
    HardwareTargetAnnotation indicating which physical device and execution
    stream the dispatch layer has assigned for that kernel's execution. The
    annotation captures the device identifier, vendor name (e.g. "nvidia",
    "amd"), architecture generation string (e.g. "sm_90", "gfx942"), and
    the assigned stream or queue identifier.

    These annotations are written by the hardware-aware dispatch layer and
    consumed by the scheduler for multi-stream emission and by the code
    generation bridge for per-target TTIR emission. Annotations are
    target-agnostic at KGIR construction time and populated during the
    dispatch phase.
  }];

  let parameters = (ins
    "int64_t":$device_id,
    StringRefParameter<"device vendor, e.g. nvidia or amd">:$vendor,
    StringRefParameter<"architecture generation, e.g. sm_90 or gfx942">:$arch,
    "int64_t":$stream_id
  );

  let assemblyFormat = "`<` $device_id `,` $vendor `,` $arch `,` $stream_id `>`";
}

//===----------------------------------------------------------------------===//
// RuntimePerformanceAttr
//===----------------------------------------------------------------------===//

def TTKGIR_RuntimePerformanceAttr
    : AttrDef<TritonKGIR_Dialect, "RuntimePerformance"> {
  let mnemonic = "runtime_perf";
  let attrName = "ttkgir.runtime_perf";

  let description = [{
    Measured per-target performance data for the closed-loop feedback mechanism.

    RuntimePerformance attributes are written back to KGIR node annotations
    by the runtime profiler after each execution iteration. They capture
    wall-clock execution time, achieved memory throughput, SM/CU occupancy
    ratio, kernel launch overhead, the device on which the measurement was
    taken, and the feedback iteration number.

    The feedback controller compares predicted performance (from the
    adaptive cost model) against these measured values. When prediction
    error exceeds the configurable sensitivity threshold
    (TRITON_FEEDBACK_SENSITIVITY), re-optimization is triggered. The
    iteration field enables tracking measurement history across feedback
    iterations for convergence detection (decision changes < 2% across
    consecutive iterations).

    Occupancy is expressed as a ratio in [0.0, 1.0] representing the
    fraction of maximum theoretical occupancy achieved.
  }];

  let parameters = (ins
    "double":$wall_clock_us,
    "double":$memory_throughput_gbps,
    "double":$occupancy,
    "double":$launch_overhead_us,
    "int64_t":$target_device_id,
    "int64_t":$iteration
  );

  let assemblyFormat = [{
    `<` $wall_clock_us `,` $memory_throughput_gbps `,` $occupancy `,`
        $launch_overhead_us `,` $target_device_id `,` $iteration `>`
  }];
}

//===----------------------------------------------------------------------===//
// FusionDecisionAttr
//===----------------------------------------------------------------------===//

def TTKGIR_FusionDecisionAttr
    : AttrDef<TritonKGIR_Dialect, "FusionDecision"> {
  let mnemonic = "fusion_decision";
  let attrName = "ttkgir.fusion_decision";

  let description = [{
    Fusion decision metadata recording whether a kernel pair was fused or
    rejected, the type of fusion considered, and the decision rationale.

    FusionDecision attributes are per-target because fusibility varies by
    hardware characteristics: shared memory capacity, register file size,
    and warp/wavefront width differ across GPU targets, making a fusion
    profitable on one target but unprofitable on another. The
    target_device_id field identifies which hardware target this decision
    applies to.

    The fusion_type field indicates the fusion strategy considered:
    "producer_consumer" for producer-consumer fusion (Kernel A writes a
    tensor that Kernel B reads with no other consumers), or "sibling" for
    sibling/horizontal fusion (independent kernels with compatible grid
    geometries merged into a single launch).

    The reason field provides a human-readable rationale for the decision,
    such as "single_consumer_compatible_tiling" for accepted fusions or
    "smem_budget_exceeded" for rejected ones. The estimated_speedup field
    records the cost model's predicted speedup factor (values > 1.0
    indicate expected improvement).

    During closed-loop feedback, fusion decisions may be reversed if
    measured performance contradicts the cost model prediction.
  }];

  let parameters = (ins
    DefaultValuedParameter<"bool", "false",
        "whether fusion was applied">:$is_fused,
    StringRefParameter<"fusion type: producer_consumer or sibling">:$fusion_type,
    StringRefParameter<"decision rationale or rejection reason">:$reason,
    "double":$estimated_speedup,
    "int64_t":$target_device_id
  );

  let assemblyFormat = [{
    `<` $is_fused `,` $fusion_type `,` $reason `,`
        $estimated_speedup `,` $target_device_id `>`
  }];
}

#endif // TRITONKGIR_ATTRDEFS
