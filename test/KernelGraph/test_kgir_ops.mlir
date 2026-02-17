// RUN: triton-opt %s | FileCheck %s
//
// FileCheck-based MLIR lit tests for the TritonKGIR dialect.
// Verifies correct parsing, printing, and round-trip of all 6 KGIR
// operations (kernel_launch, data_dep, anti_dep, transfer, fused_kernel,
// graph) and their associated custom types and attributes.
//
// NOTE: MLIR generic format prints properties in alphabetical order.
// All CHECK-SAME patterns follow alphabetical property name ordering.

// ===----------------------------------------------------------------------===
// Test ttkgir.kernel_launch: Required attributes only
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: grid_dims = array<i64: 128, 1, 1>
// CHECK-SAME: kernel_name = "test_basic_launch"
// CHECK-SAME: node_id = 0 : i32
// CHECK-SAME: num_warps = 4 : i32
// CHECK-SAME: register_pressure = 64 : i32
// CHECK-SAME: shared_memory_bytes = 16384 : i32
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 128, 1, 1>,
  kernel_name = "test_basic_launch",
  node_id = 0 : i32,
  num_warps = 4 : i32,
  register_pressure = 64 : i32,
  shared_memory_bytes = 16384 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.kernel_launch: With optional tensor_shapes attribute
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: grid_dims = array<i64: 64, 4, 1>
// CHECK-SAME: kernel_name = "test_launch_with_shapes"
// CHECK-SAME: node_id = 1 : i32
// CHECK-SAME: num_warps = 8 : i32
// CHECK-SAME: register_pressure = 48 : i32
// CHECK-SAME: shared_memory_bytes = 49152 : i32
// CHECK-SAME: tensor_shapes = array<i64: 1024, 1024, 512>
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 64, 4, 1>,
  kernel_name = "test_launch_with_shapes",
  node_id = 1 : i32,
  num_warps = 8 : i32,
  register_pressure = 48 : i32,
  shared_memory_bytes = 49152 : i32,
  tensor_shapes = array<i64: 1024, 1024, 512>
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.kernel_launch: With memory_access_patterns custom attribute
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: grid_dims = array<i64: 128, 1, 1>
// CHECK-SAME: kernel_name = "test_launch_mem_pattern"
// CHECK-SAME: memory_access_patterns = #ttkgir.mem_access_pattern<"readwrite", 0, true, 4194304>
// CHECK-SAME: node_id = 2 : i32
// CHECK-SAME: num_warps = 4 : i32
// CHECK-SAME: register_pressure = 64 : i32
// CHECK-SAME: shared_memory_bytes = 16384 : i32
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 128, 1, 1>,
  kernel_name = "test_launch_mem_pattern",
  memory_access_patterns = #ttkgir.mem_access_pattern<"readwrite", 0, true, 4194304>,
  node_id = 2 : i32,
  num_warps = 4 : i32,
  register_pressure = 64 : i32,
  shared_memory_bytes = 16384 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.kernel_launch: With hardware_target custom attribute
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: grid_dims = array<i64: 128, 1, 1>
// CHECK-SAME: hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>
// CHECK-SAME: kernel_name = "test_launch_hw_target"
// CHECK-SAME: node_id = 3 : i32
// CHECK-SAME: num_warps = 4 : i32
// CHECK-SAME: register_pressure = 64 : i32
// CHECK-SAME: shared_memory_bytes = 16384 : i32
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 128, 1, 1>,
  hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>,
  kernel_name = "test_launch_hw_target",
  node_id = 3 : i32,
  num_warps = 4 : i32,
  register_pressure = 64 : i32,
  shared_memory_bytes = 16384 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.kernel_launch: With runtime_perf custom attribute
// (Partial match on runtime_perf to avoid float format sensitivity)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: grid_dims = array<i64: 128, 1, 1>
// CHECK-SAME: kernel_name = "test_launch_runtime_perf"
// CHECK-SAME: node_id = 4 : i32
// CHECK-SAME: num_warps = 4 : i32
// CHECK-SAME: register_pressure = 64 : i32
// CHECK-SAME: runtime_perf = #ttkgir.runtime_perf<
// CHECK-SAME: shared_memory_bytes = 16384 : i32
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 128, 1, 1>,
  kernel_name = "test_launch_runtime_perf",
  node_id = 4 : i32,
  num_warps = 4 : i32,
  register_pressure = 64 : i32,
  runtime_perf = #ttkgir.runtime_perf<125.5, 1200.0, 0.85, 5.5, 0, 1>,
  shared_memory_bytes = 16384 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.kernel_launch: Comprehensive — ALL optional attributes present
// (AMD target with write-only non-contiguous access pattern)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: grid_dims = array<i64: 256, 2, 1>
// CHECK-SAME: hardware_target = #ttkgir.hw_target<1, "amd", "gfx942", 2>
// CHECK-SAME: kernel_name = "test_launch_all_opts"
// CHECK-SAME: memory_access_patterns = #ttkgir.mem_access_pattern<"write", 1, false, 8388608>
// CHECK-SAME: node_id = 5 : i32
// CHECK-SAME: num_warps = 8 : i32
// CHECK-SAME: register_pressure = 96 : i32
// CHECK-SAME: runtime_perf = #ttkgir.runtime_perf<
// CHECK-SAME: shared_memory_bytes = 32768 : i32
// CHECK-SAME: tensor_shapes = array<i64: 256, 256>
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 256, 2, 1>,
  hardware_target = #ttkgir.hw_target<1, "amd", "gfx942", 2>,
  kernel_name = "test_launch_all_opts",
  memory_access_patterns = #ttkgir.mem_access_pattern<"write", 1, false, 8388608>,
  node_id = 5 : i32,
  num_warps = 8 : i32,
  register_pressure = 96 : i32,
  runtime_perf = #ttkgir.runtime_perf<200.0, 800.0, 0.75, 3.5, 1, 2>,
  shared_memory_bytes = 32768 : i32,
  tensor_shapes = array<i64: 256, 256>
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.data_dep: Flow dependency (producer-consumer)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.data_dep"
// CHECK-SAME: dep_type = "flow"
// CHECK-SAME: dest_node_id = 1 : i32
// CHECK-SAME: source_node_id = 0 : i32
// CHECK-SAME: tensor_index = 0 : i32
"ttkgir.data_dep"() <{
  dep_type = "flow",
  dest_node_id = 1 : i32,
  source_node_id = 0 : i32,
  tensor_index = 0 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.data_dep: Output dependency
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.data_dep"
// CHECK-SAME: dep_type = "output"
// CHECK-SAME: dest_node_id = 2 : i32
// CHECK-SAME: source_node_id = 1 : i32
// CHECK-SAME: tensor_index = 1 : i32
"ttkgir.data_dep"() <{
  dep_type = "output",
  dest_node_id = 2 : i32,
  source_node_id = 1 : i32,
  tensor_index = 1 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.data_dep: Input dependency
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.data_dep"
// CHECK-SAME: dep_type = "input"
// CHECK-SAME: dest_node_id = 2 : i32
// CHECK-SAME: source_node_id = 0 : i32
// CHECK-SAME: tensor_index = 0 : i32
"ttkgir.data_dep"() <{
  dep_type = "input",
  dest_node_id = 2 : i32,
  source_node_id = 0 : i32,
  tensor_index = 0 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.anti_dep: Write-after-read (WAR) conflict
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.anti_dep"
// CHECK-SAME: conflict_type = "war"
// CHECK-SAME: dest_node_id = 2 : i32
// CHECK-SAME: source_node_id = 0 : i32
// CHECK-SAME: tensor_index = 1 : i32
"ttkgir.anti_dep"() <{
  conflict_type = "war",
  dest_node_id = 2 : i32,
  source_node_id = 0 : i32,
  tensor_index = 1 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.anti_dep: Write-after-write (WAW) conflict
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.anti_dep"
// CHECK-SAME: conflict_type = "waw"
// CHECK-SAME: dest_node_id = 3 : i32
// CHECK-SAME: source_node_id = 1 : i32
// CHECK-SAME: tensor_index = 0 : i32
"ttkgir.anti_dep"() <{
  conflict_type = "waw",
  dest_node_id = 3 : i32,
  source_node_id = 1 : i32,
  tensor_index = 0 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.transfer: Cross-device PCIe transfer (no optional attrs)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.transfer"
// CHECK-SAME: dest_device_id = 1 : i32
// CHECK-SAME: dest_node_id = 3 : i32
// CHECK-SAME: interconnect_type = "pcie_4"
// CHECK-SAME: source_device_id = 0 : i32
// CHECK-SAME: source_node_id = 0 : i32
// CHECK-SAME: tensor_index = 0 : i32
// CHECK-SAME: transfer_size_bytes = 8388608 : i64
"ttkgir.transfer"() <{
  dest_device_id = 1 : i32,
  dest_node_id = 3 : i32,
  interconnect_type = "pcie_4",
  source_device_id = 0 : i32,
  source_node_id = 0 : i32,
  tensor_index = 0 : i32,
  transfer_size_bytes = 8388608 : i64
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.transfer: NVLink with optional estimated_latency_us
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.transfer"
// CHECK-SAME: dest_device_id = 1 : i32
// CHECK-SAME: dest_node_id = 4 : i32
// CHECK-SAME: estimated_latency_us = 5.000000e+02 : f64
// CHECK-SAME: interconnect_type = "nvlink_4"
// CHECK-SAME: source_device_id = 0 : i32
// CHECK-SAME: source_node_id = 1 : i32
// CHECK-SAME: tensor_index = 2 : i32
// CHECK-SAME: transfer_size_bytes = 67108864 : i64
"ttkgir.transfer"() <{
  dest_device_id = 1 : i32,
  dest_node_id = 4 : i32,
  estimated_latency_us = 5.000000e+02 : f64,
  interconnect_type = "nvlink_4",
  source_device_id = 0 : i32,
  source_node_id = 1 : i32,
  tensor_index = 2 : i32,
  transfer_size_bytes = 67108864 : i64
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.fused_kernel: Producer-consumer fusion (required attrs only)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.fused_kernel"
// CHECK-SAME: combined_grid_dims = array<i64: 128, 1, 1>
// CHECK-SAME: combined_register_pressure = 96 : i32
// CHECK-SAME: combined_shared_memory_bytes = 32768 : i32
// CHECK-SAME: fused_name = "test_fused_matmul_relu"
// CHECK-SAME: fused_node_ids = array<i32: 0, 1>
// CHECK-SAME: fusion_type = "producer_consumer"
// CHECK-SAME: node_id = 10 : i32
// CHECK-SAME: target_device_id = 0 : i32
"ttkgir.fused_kernel"() <{
  combined_grid_dims = array<i64: 128, 1, 1>,
  combined_register_pressure = 96 : i32,
  combined_shared_memory_bytes = 32768 : i32,
  fused_name = "test_fused_matmul_relu",
  fused_node_ids = array<i32: 0, 1>,
  fusion_type = "producer_consumer",
  node_id = 10 : i32,
  target_device_id = 0 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.fused_kernel: Sibling fusion with 3 source nodes
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.fused_kernel"
// CHECK-SAME: combined_grid_dims = array<i64: 64, 4, 1>
// CHECK-SAME: combined_register_pressure = 64 : i32
// CHECK-SAME: combined_shared_memory_bytes = 16384 : i32
// CHECK-SAME: fused_name = "test_fused_sibling_pair"
// CHECK-SAME: fused_node_ids = array<i32: 2, 3, 4>
// CHECK-SAME: fusion_type = "sibling"
// CHECK-SAME: node_id = 11 : i32
// CHECK-SAME: target_device_id = 1 : i32
"ttkgir.fused_kernel"() <{
  combined_grid_dims = array<i64: 64, 4, 1>,
  combined_register_pressure = 64 : i32,
  combined_shared_memory_bytes = 16384 : i32,
  fused_name = "test_fused_sibling_pair",
  fused_node_ids = array<i32: 2, 3, 4>,
  fusion_type = "sibling",
  node_id = 11 : i32,
  target_device_id = 1 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.fused_kernel: ALL optional attrs (fusion_decision, hw_target, runtime_perf)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.fused_kernel"
// CHECK-SAME: combined_grid_dims = array<i64: 128, 1, 1>
// CHECK-SAME: combined_register_pressure = 96 : i32
// CHECK-SAME: combined_shared_memory_bytes = 32768 : i32
// CHECK-SAME: fused_name = "test_fused_all_opts"
// CHECK-SAME: fused_node_ids = array<i32: 0, 1>
// CHECK-SAME: fusion_decision = #ttkgir.fusion_decision<true, "producer_consumer", "single_consumer_compatible_tiling",
// CHECK-SAME: fusion_type = "producer_consumer"
// CHECK-SAME: hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>
// CHECK-SAME: node_id = 12 : i32
// CHECK-SAME: runtime_perf = #ttkgir.runtime_perf<
// CHECK-SAME: target_device_id = 0 : i32
"ttkgir.fused_kernel"() <{
  combined_grid_dims = array<i64: 128, 1, 1>,
  combined_register_pressure = 96 : i32,
  combined_shared_memory_bytes = 32768 : i32,
  fused_name = "test_fused_all_opts",
  fused_node_ids = array<i32: 0, 1>,
  fusion_decision = #ttkgir.fusion_decision<true, "producer_consumer", "single_consumer_compatible_tiling", 1.5, 0>,
  fusion_type = "producer_consumer",
  hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>,
  node_id = 12 : i32,
  runtime_perf = #ttkgir.runtime_perf<100.0, 1500.0, 0.9, 4.0, 0, 3>,
  target_device_id = 0 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.graph: Empty graph with required attributes only
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.graph"
// CHECK-SAME: dispatch_mode = "balanced"
// CHECK-SAME: graph_name = "test_basic_graph"
// CHECK-SAME: hardware_profiles = "[]"
// CHECK-SAME: num_edges = 0 : i32
// CHECK-SAME: num_nodes = 0 : i32
"ttkgir.graph"() <{
  dispatch_mode = "balanced",
  graph_name = "test_basic_graph",
  hardware_profiles = "[]",
  num_edges = 0 : i32,
  num_nodes = 0 : i32
}> ({
}) : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.graph: Converged with is_converged=true and iteration_count
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.graph"
// CHECK-SAME: dispatch_mode = "performance"
// CHECK-SAME: graph_name = "test_converged_graph"
// CHECK-SAME: hardware_profiles = "[{
// CHECK-SAME: is_converged = true
// CHECK-SAME: iteration_count = 7 : i32
// CHECK-SAME: num_edges = 3 : i32
// CHECK-SAME: num_nodes = 4 : i32
"ttkgir.graph"() <{
  dispatch_mode = "performance",
  graph_name = "test_converged_graph",
  hardware_profiles = "[{\"vendor\":\"nvidia\",\"arch\":\"sm_90\"}]",
  is_converged = true,
  iteration_count = 7 : i32,
  num_edges = 3 : i32,
  num_nodes = 4 : i32
}> ({
}) : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.graph: Unconverged with is_converged=false and max iterations
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.graph"
// CHECK-SAME: dispatch_mode = "cost"
// CHECK-SAME: graph_name = "test_unconverged_graph"
// CHECK-SAME: hardware_profiles = "[]"
// CHECK-SAME: is_converged = false
// CHECK-SAME: iteration_count = 20 : i32
// CHECK-SAME: num_edges = 1 : i32
// CHECK-SAME: num_nodes = 2 : i32
"ttkgir.graph"() <{
  dispatch_mode = "cost",
  graph_name = "test_unconverged_graph",
  hardware_profiles = "[]",
  is_converged = false,
  iteration_count = 20 : i32,
  num_edges = 1 : i32,
  num_nodes = 2 : i32
}> ({
}) : () -> ()

// ===----------------------------------------------------------------------===
// Test ttkgir.graph: With child kernel_launch operations in region
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.graph"
// CHECK-SAME: dispatch_mode = "balanced"
// CHECK-SAME: graph_name = "test_graph_with_kernels"
// CHECK-SAME: num_edges = 0 : i32
// CHECK-SAME: num_nodes = 2 : i32
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: kernel_name = "matmul_kernel"
// CHECK-SAME: node_id = 0 : i32
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: kernel_name = "relu_kernel"
// CHECK-SAME: node_id = 1 : i32
"ttkgir.graph"() <{
  dispatch_mode = "balanced",
  graph_name = "test_graph_with_kernels",
  hardware_profiles = "[]",
  num_edges = 0 : i32,
  num_nodes = 2 : i32
}> ({
  "ttkgir.kernel_launch"() <{
    grid_dims = array<i64: 128, 1, 1>,
    kernel_name = "matmul_kernel",
    node_id = 0 : i32,
    num_warps = 4 : i32,
    register_pressure = 48 : i32,
    shared_memory_bytes = 16384 : i32
  }> : () -> ()
  "ttkgir.kernel_launch"() <{
    grid_dims = array<i64: 128, 1, 1>,
    kernel_name = "relu_kernel",
    node_id = 1 : i32,
    num_warps = 4 : i32,
    register_pressure = 16 : i32,
    shared_memory_bytes = 0 : i32
  }> : () -> ()
}) : () -> ()

// ===----------------------------------------------------------------------===
// Integration Test: Complete transformer graph with all edge types
// (3 kernels, 2 data deps, 1 anti dep, 1 cross-device transfer)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.graph"
// CHECK-SAME: dispatch_mode = "performance"
// CHECK-SAME: graph_name = "test_complete_transformer_graph"
// CHECK-SAME: num_edges = 3 : i32
// CHECK-SAME: num_nodes = 3 : i32
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: kernel_name = "attention_matmul"
// CHECK-SAME: memory_access_patterns = #ttkgir.mem_access_pattern<"readwrite", 0, true, 4194304>
// CHECK-SAME: node_id = 0 : i32
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: kernel_name = "add_bias"
// CHECK-SAME: node_id = 1 : i32
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: hardware_target = #ttkgir.hw_target<1, "nvidia", "sm_80", 0>
// CHECK-SAME: kernel_name = "layernorm"
// CHECK-SAME: node_id = 2 : i32
// CHECK: "ttkgir.data_dep"
// CHECK-SAME: dep_type = "flow"
// CHECK-SAME: dest_node_id = 1 : i32
// CHECK-SAME: source_node_id = 0 : i32
// CHECK: "ttkgir.data_dep"
// CHECK-SAME: dep_type = "flow"
// CHECK-SAME: dest_node_id = 2 : i32
// CHECK-SAME: source_node_id = 1 : i32
// CHECK: "ttkgir.anti_dep"
// CHECK-SAME: conflict_type = "war"
// CHECK-SAME: dest_node_id = 2 : i32
// CHECK-SAME: source_node_id = 0 : i32
// CHECK: "ttkgir.transfer"
// CHECK-SAME: dest_device_id = 1 : i32
// CHECK-SAME: interconnect_type = "nvlink_4"
// CHECK-SAME: source_device_id = 0 : i32
// CHECK-SAME: transfer_size_bytes = 4194304 : i64
"ttkgir.graph"() <{
  dispatch_mode = "performance",
  graph_name = "test_complete_transformer_graph",
  hardware_profiles = "[{\"vendor\":\"nvidia\",\"arch\":\"sm_90\"},{\"vendor\":\"nvidia\",\"arch\":\"sm_80\"}]",
  num_edges = 3 : i32,
  num_nodes = 3 : i32
}> ({
  "ttkgir.kernel_launch"() <{
    grid_dims = array<i64: 128, 4, 1>,
    kernel_name = "attention_matmul",
    memory_access_patterns = #ttkgir.mem_access_pattern<"readwrite", 0, true, 4194304>,
    node_id = 0 : i32,
    num_warps = 8 : i32,
    register_pressure = 64 : i32,
    shared_memory_bytes = 32768 : i32
  }> : () -> ()
  "ttkgir.kernel_launch"() <{
    grid_dims = array<i64: 128, 1, 1>,
    kernel_name = "add_bias",
    node_id = 1 : i32,
    num_warps = 4 : i32,
    register_pressure = 24 : i32,
    shared_memory_bytes = 0 : i32
  }> : () -> ()
  "ttkgir.kernel_launch"() <{
    grid_dims = array<i64: 128, 1, 1>,
    hardware_target = #ttkgir.hw_target<1, "nvidia", "sm_80", 0>,
    kernel_name = "layernorm",
    node_id = 2 : i32,
    num_warps = 4 : i32,
    register_pressure = 32 : i32,
    shared_memory_bytes = 8192 : i32
  }> : () -> ()
  "ttkgir.data_dep"() <{
    dep_type = "flow",
    dest_node_id = 1 : i32,
    source_node_id = 0 : i32,
    tensor_index = 0 : i32
  }> : () -> ()
  "ttkgir.data_dep"() <{
    dep_type = "flow",
    dest_node_id = 2 : i32,
    source_node_id = 1 : i32,
    tensor_index = 0 : i32
  }> : () -> ()
  "ttkgir.anti_dep"() <{
    conflict_type = "war",
    dest_node_id = 2 : i32,
    source_node_id = 0 : i32,
    tensor_index = 1 : i32
  }> : () -> ()
  "ttkgir.transfer"() <{
    dest_device_id = 1 : i32,
    dest_node_id = 2 : i32,
    interconnect_type = "nvlink_4",
    source_device_id = 0 : i32,
    source_node_id = 1 : i32,
    tensor_index = 0 : i32,
    transfer_size_bytes = 4194304 : i64
  }> : () -> ()
}) : () -> ()

// ===----------------------------------------------------------------------===
// Attribute Round-Trip: memory_access_patterns — read, contiguous
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: kernel_name = "test_attr_mem_read"
// CHECK-SAME: memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 1048576>
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 64, 1, 1>,
  kernel_name = "test_attr_mem_read",
  memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 1048576>,
  node_id = 20 : i32,
  num_warps = 4 : i32,
  register_pressure = 32 : i32,
  shared_memory_bytes = 8192 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Attribute Round-Trip: memory_access_patterns — write, non-contiguous
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: kernel_name = "test_attr_mem_write_noncontig"
// CHECK-SAME: memory_access_patterns = #ttkgir.mem_access_pattern<"write", 2, false, 2097152>
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 64, 1, 1>,
  kernel_name = "test_attr_mem_write_noncontig",
  memory_access_patterns = #ttkgir.mem_access_pattern<"write", 2, false, 2097152>,
  node_id = 21 : i32,
  num_warps = 4 : i32,
  register_pressure = 32 : i32,
  shared_memory_bytes = 8192 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Attribute Round-Trip: hardware_target — NVIDIA A100 (sm_80)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_80", 0>
// CHECK-SAME: kernel_name = "test_attr_hw_nvidia_a100"
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 128, 1, 1>,
  hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_80", 0>,
  kernel_name = "test_attr_hw_nvidia_a100",
  node_id = 22 : i32,
  num_warps = 4 : i32,
  register_pressure = 64 : i32,
  shared_memory_bytes = 16384 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Attribute Round-Trip: hardware_target — AMD MI300X (gfx942)
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: hardware_target = #ttkgir.hw_target<2, "amd", "gfx942", 1>
// CHECK-SAME: kernel_name = "test_attr_hw_amd_mi300"
"ttkgir.kernel_launch"() <{
  grid_dims = array<i64: 128, 1, 1>,
  hardware_target = #ttkgir.hw_target<2, "amd", "gfx942", 1>,
  kernel_name = "test_attr_hw_amd_mi300",
  node_id = 23 : i32,
  num_warps = 4 : i32,
  register_pressure = 64 : i32,
  shared_memory_bytes = 16384 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Attribute Round-Trip: fusion_decision — accepted fusion
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.fused_kernel"
// CHECK-SAME: fused_name = "test_fused_decision_accepted"
// CHECK-SAME: fusion_decision = #ttkgir.fusion_decision<true, "producer_consumer", "eliminated_global_roundtrip",
// CHECK-SAME: fusion_type = "producer_consumer"
// CHECK-SAME: node_id = 30 : i32
"ttkgir.fused_kernel"() <{
  combined_grid_dims = array<i64: 128, 1, 1>,
  combined_register_pressure = 80 : i32,
  combined_shared_memory_bytes = 32768 : i32,
  fused_name = "test_fused_decision_accepted",
  fused_node_ids = array<i32: 0, 1>,
  fusion_decision = #ttkgir.fusion_decision<true, "producer_consumer", "eliminated_global_roundtrip", 1.35, 0>,
  fusion_type = "producer_consumer",
  node_id = 30 : i32,
  target_device_id = 0 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Attribute Round-Trip: fusion_decision — rejected fusion
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.fused_kernel"
// CHECK-SAME: fused_name = "test_fused_decision_rejected"
// CHECK-SAME: fusion_decision = #ttkgir.fusion_decision<false, "sibling", "resource_limit_exceeded",
// CHECK-SAME: fusion_type = "sibling"
// CHECK-SAME: node_id = 31 : i32
"ttkgir.fused_kernel"() <{
  combined_grid_dims = array<i64: 256, 1, 1>,
  combined_register_pressure = 128 : i32,
  combined_shared_memory_bytes = 65536 : i32,
  fused_name = "test_fused_decision_rejected",
  fused_node_ids = array<i32: 5, 6>,
  fusion_decision = #ttkgir.fusion_decision<false, "sibling", "resource_limit_exceeded", 0.8, 0>,
  fusion_type = "sibling",
  node_id = 31 : i32,
  target_device_id = 0 : i32
}> : () -> ()

// ===----------------------------------------------------------------------===
// Integration Test: Fused transformer graph (post-optimization state)
// Graph with fused kernel, unfused kernel, and data dependency
// ===----------------------------------------------------------------------===
// CHECK: "ttkgir.graph"
// CHECK-SAME: dispatch_mode = "balanced"
// CHECK-SAME: graph_name = "test_fused_transformer_graph"
// CHECK-SAME: is_converged = true
// CHECK-SAME: iteration_count = 5 : i32
// CHECK-SAME: num_edges = 1 : i32
// CHECK-SAME: num_nodes = 2 : i32
// CHECK: "ttkgir.fused_kernel"
// CHECK-SAME: fused_name = "fused_attention_block"
// CHECK-SAME: fusion_decision = #ttkgir.fusion_decision<true, "producer_consumer", "single_consumer_compatible_tiling",
// CHECK-SAME: hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>
// CHECK-SAME: node_id = 100 : i32
// CHECK-SAME: runtime_perf = #ttkgir.runtime_perf<
// CHECK: "ttkgir.kernel_launch"
// CHECK-SAME: hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>
// CHECK-SAME: kernel_name = "output_projection"
// CHECK-SAME: node_id = 2 : i32
// CHECK: "ttkgir.data_dep"
// CHECK-SAME: dep_type = "flow"
// CHECK-SAME: dest_node_id = 2 : i32
// CHECK-SAME: source_node_id = 100 : i32
"ttkgir.graph"() <{
  dispatch_mode = "balanced",
  graph_name = "test_fused_transformer_graph",
  hardware_profiles = "[{\"vendor\":\"nvidia\",\"arch\":\"sm_90\"}]",
  is_converged = true,
  iteration_count = 5 : i32,
  num_edges = 1 : i32,
  num_nodes = 2 : i32
}> ({
  "ttkgir.fused_kernel"() <{
    combined_grid_dims = array<i64: 128, 4, 1>,
    combined_register_pressure = 80 : i32,
    combined_shared_memory_bytes = 49152 : i32,
    fused_name = "fused_attention_block",
    fused_node_ids = array<i32: 0, 1>,
    fusion_decision = #ttkgir.fusion_decision<true, "producer_consumer", "single_consumer_compatible_tiling", 1.45, 0>,
    fusion_type = "producer_consumer",
    hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>,
    node_id = 100 : i32,
    runtime_perf = #ttkgir.runtime_perf<95.0, 1800.0, 0.92, 4.5, 0, 5>,
    target_device_id = 0 : i32
  }> : () -> ()
  "ttkgir.kernel_launch"() <{
    grid_dims = array<i64: 64, 2, 1>,
    hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>,
    kernel_name = "output_projection",
    node_id = 2 : i32,
    num_warps = 4 : i32,
    register_pressure = 48 : i32,
    shared_memory_bytes = 16384 : i32
  }> : () -> ()
  "ttkgir.data_dep"() <{
    dep_type = "flow",
    dest_node_id = 2 : i32,
    source_node_id = 100 : i32,
    tensor_index = 0 : i32
  }> : () -> ()
}) : () -> ()
