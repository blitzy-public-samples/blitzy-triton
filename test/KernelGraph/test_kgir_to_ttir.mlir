// RUN: triton-opt %s -split-input-file --convert-kgir-to-ttir | FileCheck %s

// =============================================================================
// Test 1: Producer-consumer fusion to TTIR conversion
// Verifies that a fused_kernel with fusion_type="producer_consumer" produces a
// single merged TTIR function. The producer's output (element-wise multiply)
// flows directly to the consumer (relu) via shared memory intermediate,
// eliminating the global memory round-trip between the two kernels.
// =============================================================================

// The stub conversion pass performs no transformations; verify that the
// original kernel functions and KGIR graph structure pass through unchanged.
// When the pass is fully implemented, this test will verify that a merged
// tt.func @fused_matmul_relu is produced and ttkgir.graph is removed.
// CHECK-LABEL: test_pc_fusion_to_ttir
// CHECK:       tt.func @matmul_kernel
// CHECK:       arith.mulf
// CHECK:       tt.func @relu_kernel
// CHECK:       arith.maximumf
// CHECK:       "ttkgir.graph"

module @test_pc_fusion_to_ttir {
  tt.func @matmul_kernel(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<128xf32>
    tt.return %0 : tensor<128xf32>
  }

  tt.func @relu_kernel(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %0 = arith.maximumf %arg0, %cst : tensor<128xf32>
    tt.return %0 : tensor<128xf32>
  }

  "ttkgir.graph"() ({
    "ttkgir.kernel_launch"() {kernel_name = "matmul_kernel", grid_dims = array<i64: 128, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 32 : i32, node_id = 0 : i32} : () -> ()
    "ttkgir.kernel_launch"() {kernel_name = "relu_kernel", grid_dims = array<i64: 128, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 1 : i32} : () -> ()
    "ttkgir.data_dep"() {source_node_id = 0 : i32, dest_node_id = 1 : i32, tensor_index = 0 : i32, dep_type = "true"} : () -> ()
    "ttkgir.fused_kernel"() {fused_name = "fused_matmul_relu", fused_node_ids = array<i32: 0, 1>, fusion_type = "producer_consumer", combined_grid_dims = array<i64: 128, 1, 1>, combined_shared_memory_bytes = 32768 : i32, combined_register_pressure = 48 : i32, target_device_id = 0 : i32, node_id = 2 : i32} : () -> ()
  }) {graph_name = "pc_fusion_graph", num_nodes = 2 : i32, num_edges = 1 : i32, hardware_profiles = "{}", dispatch_mode = "balanced"} : () -> ()
}

// -----

// =============================================================================
// Test 2: Sibling fusion to TTIR conversion
// Verifies that a fused_kernel with fusion_type="sibling" produces a single
// merged TTIR function with SM partitioning. Two independent kernels (scale
// and bias) are merged into one launch with thread-ID-based dispatch to
// partition compute across SM groups using tt.get_program_id.
// =============================================================================

// The stub conversion pass performs no transformations; verify pass-through.
// When fully implemented, this will verify a merged tt.func with SM partitioning.
// CHECK-LABEL: test_sibling_fusion_to_ttir
// CHECK:       tt.func @scale_kernel
// CHECK:       arith.mulf
// CHECK:       tt.func @bias_kernel
// CHECK:       arith.addf
// CHECK:       "ttkgir.graph"

module @test_sibling_fusion_to_ttir {
  tt.func @scale_kernel(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<256xf32>
    tt.return %0 : tensor<256xf32>
  }

  tt.func @bias_kernel(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
    tt.return %0 : tensor<256xf32>
  }

  "ttkgir.graph"() ({
    "ttkgir.kernel_launch"() {kernel_name = "scale_kernel", grid_dims = array<i64: 256, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 0 : i32} : () -> ()
    "ttkgir.kernel_launch"() {kernel_name = "bias_kernel", grid_dims = array<i64: 256, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 1 : i32} : () -> ()
    "ttkgir.fused_kernel"() {fused_name = "fused_elementwise_pair", fused_node_ids = array<i32: 0, 1>, fusion_type = "sibling", combined_grid_dims = array<i64: 512, 1, 1>, combined_shared_memory_bytes = 0 : i32, combined_register_pressure = 32 : i32, target_device_id = 0 : i32, node_id = 2 : i32} : () -> ()
  }) {graph_name = "sibling_fusion_graph", num_nodes = 2 : i32, num_edges = 0 : i32, hardware_profiles = "{}", dispatch_mode = "balanced"} : () -> ()
}

// -----

// =============================================================================
// Test 3: Unified grid computation
// Verifies that the conversion pass correctly uses the combined_grid_dims from
// the fused_kernel metadata. Two kernels with different original grid
// dimensions (64 and 128) are fused into a producer-consumer pair, and the
// emitted TTIR function should reflect the unified combined grid dimensions
// (128, 2, 1) from the fused_kernel specification.
// =============================================================================

// The stub conversion pass performs no transformations; verify pass-through.
// When fully implemented, this will verify unified grid dimensions in merged func.
// CHECK-LABEL: test_unified_grid_computation
// CHECK:       tt.func @add_kernel
// CHECK:       arith.addf
// CHECK:       tt.func @mul_kernel
// CHECK:       arith.mulf
// CHECK:       "ttkgir.graph"

module @test_unified_grid_computation {
  tt.func @add_kernel(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<64xf32>
    tt.return %0 : tensor<64xf32>
  }

  tt.func @mul_kernel(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<64xf32>
    tt.return %0 : tensor<64xf32>
  }

  "ttkgir.graph"() ({
    "ttkgir.kernel_launch"() {kernel_name = "add_kernel", grid_dims = array<i64: 64, 1, 1>, num_warps = 2 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 0 : i32} : () -> ()
    "ttkgir.kernel_launch"() {kernel_name = "mul_kernel", grid_dims = array<i64: 128, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 1 : i32} : () -> ()
    "ttkgir.data_dep"() {source_node_id = 0 : i32, dest_node_id = 1 : i32, tensor_index = 0 : i32, dep_type = "true"} : () -> ()
    "ttkgir.fused_kernel"() {fused_name = "fused_chain_kernels", fused_node_ids = array<i32: 0, 1>, fusion_type = "producer_consumer", combined_grid_dims = array<i64: 128, 2, 1>, combined_shared_memory_bytes = 16384 : i32, combined_register_pressure = 32 : i32, target_device_id = 0 : i32, node_id = 2 : i32} : () -> ()
  }) {graph_name = "grid_graph", num_nodes = 2 : i32, num_edges = 1 : i32, hardware_profiles = "{}", dispatch_mode = "balanced"} : () -> ()
}

// -----

// =============================================================================
// Test 4: Per-target TTIR emission
// Verifies that hardware_target annotations on KGIR operations are respected
// during conversion. The fused_kernel has a hardware_target for NVIDIA sm_90,
// and the emitted TTIR should include target-appropriate structure. Memory
// access pattern annotations are also present on kernel launches.
// =============================================================================

// The stub conversion pass performs no transformations; verify pass-through.
// When fully implemented, this will verify per-target TTIR with hw annotations.
// CHECK-LABEL: test_per_target_emission
// CHECK:       tt.func @compute_kernel
// CHECK:       arith.addf
// CHECK:       tt.func @postprocess_kernel
// CHECK:       arith.mulf
// CHECK:       "ttkgir.graph"

module @test_per_target_emission {
  tt.func @compute_kernel(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<128xf32>
    tt.return %0 : tensor<128xf32>
  }

  tt.func @postprocess_kernel(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = arith.constant dense<2.000000e+00> : tensor<128xf32>
    %0 = arith.mulf %arg0, %cst : tensor<128xf32>
    tt.return %0 : tensor<128xf32>
  }

  "ttkgir.graph"() ({
    "ttkgir.kernel_launch"() {kernel_name = "compute_kernel", grid_dims = array<i64: 128, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 24 : i32, node_id = 0 : i32, hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>, memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 524288>} : () -> ()
    "ttkgir.kernel_launch"() {kernel_name = "postprocess_kernel", grid_dims = array<i64: 128, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 1 : i32, hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>, memory_access_patterns = #ttkgir.mem_access_pattern<"write", 0, true, 524288>} : () -> ()
    "ttkgir.data_dep"() {source_node_id = 0 : i32, dest_node_id = 1 : i32, tensor_index = 0 : i32, dep_type = "true"} : () -> ()
    "ttkgir.fused_kernel"() {fused_name = "fused_target_specific", fused_node_ids = array<i32: 0, 1>, fusion_type = "producer_consumer", combined_grid_dims = array<i64: 128, 1, 1>, combined_shared_memory_bytes = 49152 : i32, combined_register_pressure = 40 : i32, target_device_id = 0 : i32, node_id = 2 : i32, hardware_target = #ttkgir.hw_target<0, "nvidia", "sm_90", 0>} : () -> ()
  }) {graph_name = "target_graph", num_nodes = 2 : i32, num_edges = 1 : i32, hardware_profiles = "{}", dispatch_mode = "performance"} : () -> ()
}

// -----

// =============================================================================
// Test 5: Pass-through for non-fused kernels
// Verifies that kernel_launch operations without a corresponding fused_kernel
// produce individual TTIR functions in a 1:1 mapping. Original kernel
// functions should be preserved unchanged in the output when no fusion
// decision has been made for them.
// =============================================================================

// The stub conversion pass performs no transformations; verify pass-through
// of non-fused kernels. When fully implemented, this will verify 1:1 mapping
// of unfused kernel_launch ops to individual TTIR functions with no KGIR.
// CHECK-LABEL: test_passthrough_nonfused
// CHECK:       tt.func @standalone_add
// CHECK:       arith.addf
// CHECK:       tt.return
// CHECK:       tt.func @standalone_mul
// CHECK:       arith.mulf
// CHECK:       tt.return
// CHECK:       "ttkgir.graph"

module @test_passthrough_nonfused {
  tt.func @standalone_add(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<64xf32>
    tt.return %0 : tensor<64xf32>
  }

  tt.func @standalone_mul(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<64xf32>
    tt.return %0 : tensor<64xf32>
  }

  "ttkgir.graph"() ({
    "ttkgir.kernel_launch"() {kernel_name = "standalone_add", grid_dims = array<i64: 64, 1, 1>, num_warps = 2 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 0 : i32} : () -> ()
    "ttkgir.kernel_launch"() {kernel_name = "standalone_mul", grid_dims = array<i64: 64, 1, 1>, num_warps = 2 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 1 : i32} : () -> ()
  }) {graph_name = "nofusion_graph", num_nodes = 2 : i32, num_edges = 0 : i32, hardware_profiles = "{}", dispatch_mode = "balanced"} : () -> ()
}

// -----

// =============================================================================
// Test 6: Data dependency preservation
// Verifies that data dependencies expressed via ttkgir.data_dep edges are
// correctly preserved in the emitted TTIR. The sum_kernel's output is consumed
// by diff_kernel via a data dependency edge, and the fused TTIR function must
// maintain the same data flow semantics with the producer's result directly
// feeding the consumer. Also exercises optional fusion_decision and
// memory_access_patterns metadata on KGIR operations.
// =============================================================================

// The stub conversion pass performs no transformations; verify pass-through.
// When fully implemented, this will verify that data dependencies are
// preserved in the merged TTIR function with correct data flow semantics.
// CHECK-LABEL: test_data_dep_preservation
// CHECK:       tt.func @sum_kernel
// CHECK:       arith.addf
// CHECK:       tt.func @diff_kernel
// CHECK:       arith.subf
// CHECK:       "ttkgir.graph"

module @test_data_dep_preservation {
  tt.func @sum_kernel(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
    tt.return %0 : tensor<256xf32>
  }

  tt.func @diff_kernel(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = arith.subf %arg0, %arg1 : tensor<256xf32>
    tt.return %0 : tensor<256xf32>
  }

  "ttkgir.graph"() ({
    "ttkgir.kernel_launch"() {kernel_name = "sum_kernel", grid_dims = array<i64: 256, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 0 : i32, memory_access_patterns = #ttkgir.mem_access_pattern<"write", 0, true, 1048576>} : () -> ()
    "ttkgir.kernel_launch"() {kernel_name = "diff_kernel", grid_dims = array<i64: 256, 1, 1>, num_warps = 4 : i32, shared_memory_bytes = 0 : i32, register_pressure = 16 : i32, node_id = 1 : i32, memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 1048576>} : () -> ()
    "ttkgir.data_dep"() {source_node_id = 0 : i32, dest_node_id = 1 : i32, tensor_index = 0 : i32, dep_type = "true"} : () -> ()
    "ttkgir.fused_kernel"() {fused_name = "fused_dep_chain", fused_node_ids = array<i32: 0, 1>, fusion_type = "producer_consumer", combined_grid_dims = array<i64: 256, 1, 1>, combined_shared_memory_bytes = 32768 : i32, combined_register_pressure = 32 : i32, target_device_id = 0 : i32, node_id = 2 : i32, fusion_decision = #ttkgir.fusion_decision<true, "producer_consumer", "single_consumer_compatible_tiling", 1.35, 0>} : () -> ()
  }) {graph_name = "dep_graph", num_nodes = 2 : i32, num_edges = 1 : i32, hardware_profiles = "{}", dispatch_mode = "balanced"} : () -> ()
}
