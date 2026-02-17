#ifndef TRITONKGIR_TYPES
#define TRITONKGIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonKGIR/IR/TritonKGIRDialect.td"

//===----------------------------------------------------------------------===//
// TTKGIR Base Type Class
//===----------------------------------------------------------------------===//

// Base class for all TritonKGIR dialect type definitions.
// Follows the TritonGPU TTG_TypeDef pattern with dialect-scoped mnemonic.
class TTKGIR_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonKGIR_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// HardwareProfileType
//===----------------------------------------------------------------------===//

// Per-device hardware profile descriptor that encapsulates compute, memory,
// and interconnect capabilities of a GPU device. Attached at compilation
// time rather than baked into the IR, enabling target-agnostic KGIR
// construction with late-binding hardware assignment.
def TTKGIR_HardwareProfileType
    : TTKGIR_TypeDef<"HardwareProfile", "hw_profile"> {
  let summary = "Hardware profile descriptor for a GPU device";
  let description = [{
    Encapsulates per-device hardware characteristics used by the
    hardware-aware dispatch layer and the adaptive cost model. Each GPU
    device in the hardware inventory is described by a HardwareProfile
    that captures its compute, memory, and interconnect capabilities.

    Hardware-specific information is encapsulated in this descriptor
    rather than being baked into the KGIR IR, enabling target-agnostic
    graph construction with late-binding hardware assignment.

    Fields correspond to the HardwareProfile concrete schema:
    - vendor: Device vendor identifier (e.g., "nvidia", "amd")
    - arch_generation: Architecture generation (e.g., "sm_90", "gfx942")
    - sm_count: Number of SMs (NVIDIA) or CUs (AMD)
    - smem_per_sm_bytes: Shared memory per SM/CU in bytes
    - registers_per_sm: Register file size per SM/CU
    - global_memory_bytes: Total device global memory in bytes
    - memory_bandwidth_gbps: Peak memory bandwidth in GB/s
    - compute_throughput_tflops: Peak compute throughput in TFLOPS
    - warp_size: Warp/wavefront width (32 for NVIDIA, 64 for AMD)
    - max_concurrent_streams: Maximum concurrent streams/queues
    - interconnect_type: Interconnect type (e.g., "pcie_4", "nvlink_4")
    - interconnect_bandwidth_gbps: Interconnect bandwidth in GB/s

    Example:
    ```mlir
    !ttkgir.hw_profile<"nvidia", "sm_90", 132, 232448, 65536,
                        85899345920, 2039.0, 82.58, 32, 128,
                        "nvlink_4", 900.0>
    ```
  }];

  let parameters = (ins
    StringRefParameter<"device vendor (e.g., nvidia, amd)">:$vendor,
    StringRefParameter<"architecture generation (e.g., sm_90, gfx942)">:$arch_generation,
    TypeParameter<"int32_t", "number of SMs (NVIDIA) or CUs (AMD)">:$sm_count,
    TypeParameter<"int32_t", "shared memory per SM/CU in bytes">:$smem_per_sm_bytes,
    TypeParameter<"int32_t", "register file size per SM/CU">:$registers_per_sm,
    TypeParameter<"int64_t", "total device global memory in bytes">:$global_memory_bytes,
    TypeParameter<"double", "peak memory bandwidth in GB/s">:$memory_bandwidth_gbps,
    TypeParameter<"double", "peak compute throughput in TFLOPS">:$compute_throughput_tflops,
    TypeParameter<"int32_t", "warp/wavefront size (32 for NVIDIA, 64 for AMD)">:$warp_size,
    TypeParameter<"int32_t", "maximum concurrent streams/queues">:$max_concurrent_streams,
    StringRefParameter<"interconnect type (e.g., pcie_4, nvlink_4)">:$interconnect_type,
    TypeParameter<"double", "interconnect bandwidth in GB/s">:$interconnect_bandwidth_gbps
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// NodeMetadataType
//===----------------------------------------------------------------------===//

// Per-kernel-node metadata capturing grid dimensions, resource usage, and
// memory access characteristics for fusion analysis and scheduling decisions.
def TTKGIR_NodeMetadataType
    : TTKGIR_TypeDef<"NodeMetadata", "node_metadata"> {
  let summary = "Per-kernel-node metadata in the kernel graph";
  let description = [{
    Captures memory access patterns, tensor shapes, grid dimensions, and
    resource usage for a kernel node in the KGIR DAG. This metadata
    drives fusion analysis decisions (resource budget checking, tiling
    compatibility) and scheduling (grid geometry matching for sibling
    fusion, register/shared memory pressure for occupancy estimation).

    Fields:
    - num_tensor_args: Number of tensor arguments to this kernel
    - grid_dim_x: Grid dimension X for the kernel launch
    - grid_dim_y: Grid dimension Y for the kernel launch
    - grid_dim_z: Grid dimension Z for the kernel launch
    - shared_memory_bytes: Shared memory usage in bytes
    - register_pressure: Estimated register usage per thread
    - num_reads: Number of global memory read operations
    - num_writes: Number of global memory write operations

    Example:
    ```mlir
    !ttkgir.node_metadata<4, 1024, 1, 1, 49152, 32, 8, 4>
    ```
  }];

  let parameters = (ins
    TypeParameter<"int32_t", "number of tensor arguments">:$num_tensor_args,
    TypeParameter<"int32_t", "grid dimension X">:$grid_dim_x,
    TypeParameter<"int32_t", "grid dimension Y">:$grid_dim_y,
    TypeParameter<"int32_t", "grid dimension Z">:$grid_dim_z,
    TypeParameter<"int32_t", "shared memory usage in bytes">:$shared_memory_bytes,
    TypeParameter<"int32_t", "estimated register usage per thread">:$register_pressure,
    TypeParameter<"int32_t", "number of global memory reads">:$num_reads,
    TypeParameter<"int32_t", "number of global memory writes">:$num_writes
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// PerformanceAnnotationType
//===----------------------------------------------------------------------===//

// Per-target measured runtime performance metrics for the closed-loop
// feedback mechanism. Written back to KGIR node annotations by the
// runtime profiler after each execution iteration.
def TTKGIR_PerformanceAnnotationType
    : TTKGIR_TypeDef<"PerformanceAnnotation", "perf_annotation"> {
  let summary = "Per-target measured runtime performance metrics";
  let description = [{
    Stores measured performance data per kernel per target for the
    closed-loop feedback mechanism. The runtime profiler writes measured
    data back into KGIR node annotations after each execution, enabling
    the feedback controller to compare predictions against actuals and
    trigger re-optimization when prediction error exceeds the configured
    sensitivity threshold (TRITON_FEEDBACK_SENSITIVITY, default 0.15).

    All metrics are doubles to accommodate fractional measurements from
    GPU event timing APIs (CUDA events, HIP events).

    Fields:
    - wall_clock_us: Wall-clock execution time in microseconds
    - memory_throughput_gbps: Measured memory throughput in GB/s
    - occupancy: SM/CU occupancy ratio (0.0 to 1.0)
    - bandwidth_utilization: Memory bandwidth utilization ratio (0.0 to 1.0)
    - launch_overhead_us: Kernel launch overhead in microseconds
    - transfer_time_us: Cross-device transfer time in microseconds

    Example:
    ```mlir
    !ttkgir.perf_annotation<125.3, 1200.5, 0.85, 0.72, 5.2, 0.0>
    ```
  }];

  let parameters = (ins
    TypeParameter<"double", "wall-clock execution time in microseconds">:$wall_clock_us,
    TypeParameter<"double", "memory throughput in GB/s">:$memory_throughput_gbps,
    TypeParameter<"double", "SM/CU occupancy ratio (0.0-1.0)">:$occupancy,
    TypeParameter<"double", "memory bandwidth utilization ratio (0.0-1.0)">:$bandwidth_utilization,
    TypeParameter<"double", "kernel launch overhead in microseconds">:$launch_overhead_us,
    TypeParameter<"double", "cross-device transfer time in microseconds">:$transfer_time_us
  );

  let hasCustomAssemblyFormat = 1;
}

#endif // TRITONKGIR_TYPES
