//===-- TritonKGIROps.td - KGIR Operation Defs -------------*- tablegen -*-===//
//
// Part of the Triton project.
//
// Defines operation types for the TritonKGIR (Kernel Graph IR) dialect.
// These operations model kernel graph semantics — kernel launch nodes,
// dependency edges, cross-device transfers, fused kernels, and a top-level
// graph container — used by the graph-level cross-kernel optimization layer.
//
// KGIR operations are structural/metadata operations representing a DAG of
// kernel launches. They do NOT model tensor computation (that is the role of
// the Triton and TritonGPU dialects). Nodes are target-agnostic at
// construction and mutable for runtime annotation write-back via the
// closed-loop feedback mechanism.
//
//===----------------------------------------------------------------------===//

#ifndef TRITONKGIR_OPS
#define TRITONKGIR_OPS

include "triton/Dialect/TritonKGIR/IR/TritonKGIRDialect.td"
include "triton/Dialect/TritonKGIR/IR/TritonKGIRTypes.td"
include "triton/Dialect/TritonKGIR/IR/TritonKGIRAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// TTKGIR Base Op Class
//===----------------------------------------------------------------------===//

// Base class for all TritonKGIR dialect operations. KGIR ops model graph
// semantics (kernel launches, dependencies, transfers, fusion results) rather
// than tensor computation, so we intentionally do NOT include
// VerifyTensorLayoutsTrait (that is for compute dialects like Triton/TritonGPU).
class TTKGIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonKGIR_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// ttkgir.kernel_launch — Single Kernel Node
//===----------------------------------------------------------------------===//

def TTKGIR_KernelLaunchOp : TTKGIR_Op<"kernel_launch", []> {
  let summary = "Represents a single kernel launch node in the kernel graph";
  let description = [{
    Models a single kernel launch within the KGIR directed acyclic graph (DAG).
    Each kernel_launch operation captures comprehensive metadata about one
    kernel invocation: its name, grid geometry, resource consumption (warps,
    shared memory, registers), memory access patterns, and optional hardware
    target and runtime performance annotations.

    This is a metadata/structural operation — it does not perform computation
    itself. It records information that drives downstream optimization passes:
    fusion analysis (checking resource budgets and tiling compatibility),
    memory planning (identifying global memory traffic), scheduling (grid
    geometry and resource pressure), and dispatch (hardware target assignment).

    The `memory_access_patterns` attribute is optional and carries per-tensor
    read/write pattern information used by the fusion analysis engine to
    identify producer-consumer pairs.

    The `hardware_target` attribute is optional and populated by the dispatch
    layer after hardware assignment. KGIR nodes are target-agnostic at
    construction time.

    The `runtime_perf` attribute is optional and written back by the runtime
    profiler during closed-loop feedback iterations. This enables the
    adaptive cost model to transition from heuristic-based (Phase 1) to
    measurement-based (Phase 2) predictions.

    The `tensor_shapes` attribute is optional and encodes the shapes of
    tensor arguments as a flat i64 array.

    The `node_id` uniquely identifies this kernel within the graph for
    cross-referencing by edge operations (data_dep, anti_dep, transfer).

    Example:
    ```mlir
    ttkgir.kernel_launch {
      kernel_name = "matmul_kernel",
      grid_dims = array<i64: 128, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 49152 : i32,
      register_pressure = 32 : i32,
      node_id = 0 : i32
    }
    ```
  }];

  let arguments = (ins
    StrAttr:$kernel_name,
    DenseI64ArrayAttr:$grid_dims,
    I32Attr:$num_warps,
    I32Attr:$shared_memory_bytes,
    I32Attr:$register_pressure,
    OptionalAttr<TTKGIR_MemoryAccessPatternAttr>:$memory_access_patterns,
    OptionalAttr<TTKGIR_HardwareTargetAnnotationAttr>:$hardware_target,
    OptionalAttr<TTKGIR_RuntimePerformanceAttr>:$runtime_perf,
    OptionalAttr<DenseI64ArrayAttr>:$tensor_shapes,
    I32Attr:$node_id
  );

  // No results — this is a metadata/structural node in the graph DAG.
  let results = (outs);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ttkgir.data_dep — Data Dependency Edge
//===----------------------------------------------------------------------===//

def TTKGIR_DataDepOp : TTKGIR_Op<"data_dep", []> {
  let summary = "Data dependency edge between kernel launch nodes";
  let description = [{
    Represents a data dependency (producer-consumer relationship) between two
    kernel nodes in the KGIR DAG. A data_dep edge indicates that the kernel
    identified by `source_node_id` produces a tensor that the kernel
    identified by `dest_node_id` consumes.

    The `tensor_index` identifies which tensor argument of the producing
    kernel corresponds to the consumed output. The `dep_type` field
    classifies the dependency for scheduling and fusion analysis:
    - "flow": Standard RAW (read-after-write) data dependency
    - "output": WAW (write-after-write) output dependency
    - "input": Shared input dependency (both kernels read the same tensor)

    Data dependency edges are used by:
    - Fusion analysis: Identifying fusible producer-consumer pairs (single
      consumer, compatible tiling, combined resources within per-SM/CU limits)
    - Scheduler: Determining legal execution orderings and critical-path
      computation
    - Memory planning: Identifying intermediate tensors eligible for
      global→shared memory promotion

    Example:
    ```mlir
    ttkgir.data_dep {
      source_node_id = 0 : i32,
      dest_node_id = 1 : i32,
      tensor_index = 0 : i32,
      dep_type = "flow"
    }
    ```
  }];

  let arguments = (ins
    I32Attr:$source_node_id,
    I32Attr:$dest_node_id,
    I32Attr:$tensor_index,
    StrAttr:$dep_type
  );

  // No results — this is a structural edge in the graph DAG.
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// ttkgir.anti_dep — Anti-Dependency Edge
//===----------------------------------------------------------------------===//

def TTKGIR_AntiDepOp : TTKGIR_Op<"anti_dep", []> {
  let summary = "Anti-dependency or resource conflict edge between kernel nodes";
  let description = [{
    Represents an anti-dependency (write-after-read), output dependency
    (write-after-write), or resource conflict between two kernel nodes in
    the KGIR DAG. Unlike data_dep edges which indicate producer-consumer
    data flow, anti_dep edges indicate ordering constraints arising from
    memory hazards or shared resource conflicts.

    The `source_node_id` and `dest_node_id` fields identify the two kernel
    nodes involved. The `tensor_index` identifies the tensor argument that
    is the source of the conflict. The `conflict_type` field classifies the
    anti-dependency:
    - "war": Write-after-read — dest writes to memory that source reads
    - "waw": Write-after-write — both nodes write to the same memory
    - "resource": Non-memory resource conflict (e.g., shared memory bank
      conflicts, register file pressure when co-scheduling)

    Anti-dependency edges are used by:
    - Scheduler: Enforcing execution ordering to prevent race conditions in
      multi-stream schedules
    - Fusion analysis: Blocking fusion of kernel pairs with conflicting
      memory access patterns

    Example:
    ```mlir
    ttkgir.anti_dep {
      source_node_id = 0 : i32,
      dest_node_id = 2 : i32,
      tensor_index = 1 : i32,
      conflict_type = "war"
    }
    ```
  }];

  let arguments = (ins
    I32Attr:$source_node_id,
    I32Attr:$dest_node_id,
    I32Attr:$tensor_index,
    StrAttr:$conflict_type
  );

  // No results — this is a structural edge in the graph DAG.
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// ttkgir.transfer — Cross-Device Data Transfer
//===----------------------------------------------------------------------===//

def TTKGIR_TransferOp : TTKGIR_Op<"transfer", []> {
  let summary = "Cross-device data transfer operation in the kernel graph";
  let description = [{
    Represents a cross-device data transfer operation inserted by the
    hardware-aware dispatch layer when the dispatch plan splits the kernel
    graph across multiple physical GPU devices.

    When the dispatch decision engine assigns producer and consumer kernels
    to different devices, transfer operations are inserted along the
    corresponding data dependency edges. Each transfer captures the source
    and destination device identifiers, the tensor being transferred, its
    size, and the interconnect mechanism used for the transfer.

    The `source_device_id` and `dest_device_id` identify the physical GPU
    devices involved. The `tensor_index` identifies which tensor argument
    is being transferred. The `transfer_size_bytes` gives the total byte
    count for the transfer. The `interconnect_type` describes the physical
    interconnect used:
    - "pcie": PCI Express (Gen 3/4/5)
    - "nvlink": NVIDIA NVLink interconnect
    - "infinity_fabric": AMD Infinity Fabric
    - "host_staging": Cross-vendor transfer via host memory staging

    The optional `estimated_latency_us` field holds the cost model's
    predicted transfer latency in microseconds, which may be refined during
    closed-loop feedback iterations.

    The `source_node_id` and `dest_node_id` identify the producing and
    consuming kernel nodes that this transfer connects.

    Example:
    ```mlir
    ttkgir.transfer {
      source_device_id = 0 : i32,
      dest_device_id = 1 : i32,
      tensor_index = 0 : i32,
      transfer_size_bytes = 67108864 : i64,
      interconnect_type = "nvlink",
      source_node_id = 0 : i32,
      dest_node_id = 3 : i32
    }
    ```
  }];

  let arguments = (ins
    I32Attr:$source_device_id,
    I32Attr:$dest_device_id,
    I32Attr:$tensor_index,
    I64Attr:$transfer_size_bytes,
    StrAttr:$interconnect_type,
    OptionalAttr<F64Attr>:$estimated_latency_us,
    I32Attr:$source_node_id,
    I32Attr:$dest_node_id
  );

  // No results — this is a structural operation in the graph DAG.
  let results = (outs);
}

//===----------------------------------------------------------------------===//
// ttkgir.fused_kernel — Fused Kernel Node
//===----------------------------------------------------------------------===//

def TTKGIR_FusedKernelOp : TTKGIR_Op<"fused_kernel", []> {
  let summary = "Represents a fused kernel combining multiple kernel launch nodes";
  let description = [{
    Models the result of kernel fusion in the KGIR DAG. A fused_kernel
    operation replaces two or more kernel_launch operations that the fusion
    analysis engine determined are profitable to merge.

    Two fusion strategies are supported:
    - "producer_consumer": Producer-consumer fusion where Kernel A writes a
      tensor that Kernel B reads with no other consumers, compatible tiling,
      and combined resources within per-SM/CU limits. The intermediate
      tensor is promoted from global memory to shared memory or registers.
    - "sibling": Sibling/horizontal fusion where independent kernels with
      compatible grid geometries are merged into a single launch with
      partitioned SM/CU allocation.

    The `fused_node_ids` array records the original kernel node IDs that
    were fused, preserving provenance for debugging and rollback during
    closed-loop feedback. The `fusion_type` classifies the fusion strategy.
    The `combined_grid_dims` gives the unified grid geometry for the fused
    kernel. The `combined_shared_memory_bytes` and
    `combined_register_pressure` reflect the aggregate resource consumption
    of the fused kernel.

    The `target_device_id` identifies the device assigned by the dispatch
    layer for executing this fused kernel.

    The optional `hardware_target` and `runtime_perf` attributes serve the
    same roles as in kernel_launch: late-bound hardware assignment and
    closed-loop performance measurement write-back.

    The optional `fusion_decision` attribute records the cost model's
    decision metadata including estimated speedup and rationale.

    The `node_id` uniquely identifies this fused kernel within the graph.

    Example:
    ```mlir
    ttkgir.fused_kernel {
      fused_name = "matmul_relu_fused",
      fused_node_ids = array<i32: 0, 1>,
      fusion_type = "producer_consumer",
      combined_grid_dims = array<i64: 128, 1, 1>,
      combined_shared_memory_bytes = 65536 : i32,
      combined_register_pressure = 48 : i32,
      target_device_id = 0 : i32,
      node_id = 10 : i32
    }
    ```
  }];

  let arguments = (ins
    StrAttr:$fused_name,
    DenseI32ArrayAttr:$fused_node_ids,
    StrAttr:$fusion_type,
    DenseI64ArrayAttr:$combined_grid_dims,
    I32Attr:$combined_shared_memory_bytes,
    I32Attr:$combined_register_pressure,
    I32Attr:$target_device_id,
    OptionalAttr<TTKGIR_HardwareTargetAnnotationAttr>:$hardware_target,
    OptionalAttr<TTKGIR_RuntimePerformanceAttr>:$runtime_perf,
    OptionalAttr<TTKGIR_FusionDecisionAttr>:$fusion_decision,
    I32Attr:$node_id
  );

  // No results — this is a metadata node in the graph DAG.
  let results = (outs);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ttkgir.graph — Top-Level Graph Container
//===----------------------------------------------------------------------===//

def TTKGIR_GraphOp : TTKGIR_Op<"graph", [
    SingleBlock,
    NoTerminator
]> {
  let summary = "Top-level container operation for the kernel graph DAG";
  let description = [{
    The graph operation is the top-level container holding the complete
    kernel graph DAG. Its single-block region contains all kernel_launch,
    data_dep, anti_dep, transfer, and fused_kernel operations that
    constitute the graph.

    The `graph_name` identifies this kernel graph (typically derived from
    the trace capture scope). The `num_nodes` and `num_edges` fields record
    the graph's size for quick introspection without traversal.

    The `hardware_profiles` field is a string attribute encoding the
    available hardware inventory as serialized JSON. This is populated by
    the trace capture mechanism at graph construction time from the
    discovered GPU devices and consumed by the dispatch layer for multi-target
    compilation and dispatch decisions.

    The `dispatch_mode` specifies the optimization objective:
    - "performance": Minimize end-to-end latency
    - "cost": Minimize total GPU-seconds (device utilization)
    - "balanced": Weighted combination of performance and cost objectives

    The optional `is_converged` flag indicates whether the closed-loop
    feedback controller has reached convergence (decision changes < 2%
    across consecutive iterations).

    The optional `iteration_count` tracks the current feedback iteration
    number (capped at TRITON_FEEDBACK_MAX_ITERS, default 20).

    The graph body region uses SingleBlock and NoTerminator traits because
    it acts as a flat container of metadata operations — there is no control
    flow within the kernel graph IR. Operations within the region encode
    graph structure through node_id cross-references rather than through
    SSA value flow.

    Example:
    ```mlir
    ttkgir.graph {
      graph_name = "transformer_block",
      num_nodes = 5 : i32,
      num_edges = 4 : i32,
      hardware_profiles = "[]",
      dispatch_mode = "balanced"
    } {
      ttkgir.kernel_launch { kernel_name = "attention", ... }
      ttkgir.kernel_launch { kernel_name = "layernorm", ... }
      ttkgir.data_dep { source_node_id = 0 : i32, dest_node_id = 1 : i32, ... }
    }
    ```
  }];

  let arguments = (ins
    StrAttr:$graph_name,
    I32Attr:$num_nodes,
    I32Attr:$num_edges,
    StrAttr:$hardware_profiles,
    StrAttr:$dispatch_mode,
    OptionalAttr<BoolAttr>:$is_converged,
    OptionalAttr<I32Attr>:$iteration_count
  );

  // Single-block region containing the kernel graph body: kernel_launch,
  // data_dep, anti_dep, transfer, and fused_kernel operations.
  let regions = (region SizedRegion<1>:$body);

  // No results — this is a top-level container.
  let results = (outs);

  let hasVerifier = 1;
}

#endif // TRITONKGIR_OPS
