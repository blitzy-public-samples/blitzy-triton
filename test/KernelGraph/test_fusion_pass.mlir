// RUN: triton-opt %s -split-input-file --ttkgir-fusion-analysis | FileCheck %s
// RUN: triton-opt %s -split-input-file --ttkgir-fusion-analysis='disable-sibling-fusion=true' | FileCheck %s --check-prefix=CHECK-NO-SIBLING

// ============================================================================
// Test 1: Producer-consumer fusion — positive case
// Two kernels: matmul (writes tensor) → relu (reads tensor, single consumer)
// Compatible grid geometries (both 128x1x1), compatible warps (both 4)
// Combined shared memory: 16384 + 0 = 16384 bytes (well within 48KB per-SM)
// Combined register pressure: max(48, 16) = reasonable
// Expected: Fusion pass creates ttkgir.fused_kernel with producer_consumer type
// ============================================================================

// CHECK-LABEL: module @test_producer_consumer_fusion_positive
// CHECK:         "ttkgir.fused_kernel"
// CHECK-SAME:      fusion_type = "producer_consumer"
// CHECK-SAME:      fused_node_ids = array<i32: 0, 1>

// Producer-consumer fusion is unaffected by disable-sibling-fusion option
// CHECK-NO-SIBLING-LABEL: module @test_producer_consumer_fusion_positive
// CHECK-NO-SIBLING:         "ttkgir.fused_kernel"
// CHECK-NO-SIBLING-SAME:      fusion_type = "producer_consumer"

module @test_producer_consumer_fusion_positive {
  "ttkgir.graph"() <{
    graph_name = "pc_fusion_test",
    num_nodes = 2 : i32,
    num_edges = 1 : i32,
    hardware_profiles = "[]",
    dispatch_mode = "performance"
  }> ({
    "ttkgir.kernel_launch"() <{
      kernel_name = "matmul",
      grid_dims = array<i64: 128, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 16384 : i32,
      register_pressure = 48 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"write", 0, true, 4194304>,
      node_id = 0 : i32
    }> : () -> ()
    "ttkgir.kernel_launch"() <{
      kernel_name = "relu",
      grid_dims = array<i64: 128, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 0 : i32,
      register_pressure = 16 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 4194304>,
      node_id = 1 : i32
    }> : () -> ()
    "ttkgir.data_dep"() <{
      source_node_id = 0 : i32,
      dest_node_id = 1 : i32,
      tensor_index = 0 : i32,
      dep_type = "flow"
    }> : () -> ()
  }) : () -> ()
}

// -----

// ============================================================================
// Test 2: Producer-consumer fusion — negative case (multiple consumers)
// Kernel A (matmul) writes tensor consumed by BOTH Kernel B (relu) and
// Kernel C (add). Violates the single-consumer requirement for
// producer-consumer fusion.
// Expected: No fused_kernel produced — multi-consumer blocks fusion
// ============================================================================

// CHECK-LABEL: module @test_producer_consumer_multi_consumer
// CHECK-NOT:     "ttkgir.fused_kernel"

module @test_producer_consumer_multi_consumer {
  "ttkgir.graph"() <{
    graph_name = "multi_consumer_test",
    num_nodes = 3 : i32,
    num_edges = 2 : i32,
    hardware_profiles = "[]",
    dispatch_mode = "performance"
  }> ({
    "ttkgir.kernel_launch"() <{
      kernel_name = "matmul",
      grid_dims = array<i64: 128, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 16384 : i32,
      register_pressure = 48 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"write", 0, true, 4194304>,
      node_id = 0 : i32
    }> : () -> ()
    "ttkgir.kernel_launch"() <{
      kernel_name = "relu",
      grid_dims = array<i64: 128, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 0 : i32,
      register_pressure = 16 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 4194304>,
      node_id = 1 : i32
    }> : () -> ()
    "ttkgir.kernel_launch"() <{
      kernel_name = "add",
      grid_dims = array<i64: 128, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 0 : i32,
      register_pressure = 16 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 4194304>,
      node_id = 2 : i32
    }> : () -> ()
    "ttkgir.data_dep"() <{
      source_node_id = 0 : i32,
      dest_node_id = 1 : i32,
      tensor_index = 0 : i32,
      dep_type = "flow"
    }> : () -> ()
    "ttkgir.data_dep"() <{
      source_node_id = 0 : i32,
      dest_node_id = 2 : i32,
      tensor_index = 0 : i32,
      dep_type = "flow"
    }> : () -> ()
  }) : () -> ()
}

// -----

// ============================================================================
// Test 3: Sibling/horizontal fusion — positive case
// Two independent kernels (elementwise_add, elementwise_mul) with no data
// dependency between them. Compatible grid geometries (both 256x1x1),
// compatible warps (both 4), combined shared memory 0+0 = 0 bytes,
// combined register pressure 24+24 = 48 (within limits).
// Expected: Fusion pass creates ttkgir.fused_kernel with sibling type
// ============================================================================

// CHECK-LABEL: module @test_sibling_fusion_positive
// CHECK:         "ttkgir.fused_kernel"
// CHECK-SAME:      fusion_type = "sibling"
// CHECK-SAME:      fused_node_ids = array<i32: 0, 1>

// With disable-sibling-fusion=true, sibling fusion must NOT occur
// CHECK-NO-SIBLING-LABEL: module @test_sibling_fusion_positive
// CHECK-NO-SIBLING-NOT:     fusion_type = "sibling"

module @test_sibling_fusion_positive {
  "ttkgir.graph"() <{
    graph_name = "sibling_fusion_test",
    num_nodes = 2 : i32,
    num_edges = 0 : i32,
    hardware_profiles = "[]",
    dispatch_mode = "performance"
  }> ({
    "ttkgir.kernel_launch"() <{
      kernel_name = "elementwise_add",
      grid_dims = array<i64: 256, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 0 : i32,
      register_pressure = 24 : i32,
      node_id = 0 : i32
    }> : () -> ()
    "ttkgir.kernel_launch"() <{
      kernel_name = "elementwise_mul",
      grid_dims = array<i64: 256, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 0 : i32,
      register_pressure = 24 : i32,
      node_id = 1 : i32
    }> : () -> ()
  }) : () -> ()
}

// -----

// ============================================================================
// Test 4: Sibling fusion — negative case (incompatible grid dimensions)
// Two independent kernels with different grid geometries:
//   - reduce_sum: grid 64x1x1 (reduction kernel, fewer blocks)
//   - elementwise_scale: grid 512x1x1 (elementwise kernel, many blocks)
// Incompatible grids prevent sibling fusion since merged launch requires
// identical grid geometry for SM/CU partitioning.
// Expected: No fused_kernel produced
// ============================================================================

// CHECK-LABEL: module @test_sibling_fusion_incompatible_grid
// CHECK-NOT:     "ttkgir.fused_kernel"

// Also no fusion when sibling is disabled (acts as scope boundary above)
// CHECK-NO-SIBLING-LABEL: module @test_sibling_fusion_incompatible_grid
// CHECK-NO-SIBLING-NOT:     "ttkgir.fused_kernel"

module @test_sibling_fusion_incompatible_grid {
  "ttkgir.graph"() <{
    graph_name = "incompatible_grid_test",
    num_nodes = 2 : i32,
    num_edges = 0 : i32,
    hardware_profiles = "[]",
    dispatch_mode = "performance"
  }> ({
    "ttkgir.kernel_launch"() <{
      kernel_name = "reduce_sum",
      grid_dims = array<i64: 64, 1, 1>,
      num_warps = 8 : i32,
      shared_memory_bytes = 4096 : i32,
      register_pressure = 32 : i32,
      node_id = 0 : i32
    }> : () -> ()
    "ttkgir.kernel_launch"() <{
      kernel_name = "elementwise_scale",
      grid_dims = array<i64: 512, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 0 : i32,
      register_pressure = 16 : i32,
      node_id = 1 : i32
    }> : () -> ()
  }) : () -> ()
}

// -----

// ============================================================================
// Test 5: Resource limit enforcement — shared memory exceeds per-SM budget
// Two kernels that form a valid producer-consumer pair (single consumer,
// compatible grids), but each uses 32768 bytes (32KB) of shared memory.
// Combined: 32768 + 32768 = 65536 bytes (64KB) which exceeds the typical
// 49152 bytes (48KB) per-SM shared memory limit on most GPU architectures.
// Expected: No fused_kernel produced — resource constraint violation
// ============================================================================

// CHECK-LABEL: module @test_resource_limit_exceeded
// CHECK-NOT:     "ttkgir.fused_kernel"

module @test_resource_limit_exceeded {
  "ttkgir.graph"() <{
    graph_name = "resource_limit_test",
    num_nodes = 2 : i32,
    num_edges = 1 : i32,
    hardware_profiles = "[]",
    dispatch_mode = "performance"
  }> ({
    "ttkgir.kernel_launch"() <{
      kernel_name = "conv2d",
      grid_dims = array<i64: 64, 64, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 32768 : i32,
      register_pressure = 64 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"write", 0, true, 16777216>,
      node_id = 0 : i32
    }> : () -> ()
    "ttkgir.kernel_launch"() <{
      kernel_name = "batchnorm",
      grid_dims = array<i64: 64, 64, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 32768 : i32,
      register_pressure = 48 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 16777216>,
      node_id = 1 : i32
    }> : () -> ()
    "ttkgir.data_dep"() <{
      source_node_id = 0 : i32,
      dest_node_id = 1 : i32,
      tensor_index = 0 : i32,
      dep_type = "flow"
    }> : () -> ()
  }) : () -> ()
}

// -----

// ============================================================================
// Test 6: Anti-dependency blocks producer-consumer fusion
// Kernel A writes tensor that Kernel B reads (flow dependency, fusible pair),
// but there is also a WAR (write-after-read) anti-dependency from A to B
// on a different tensor. The anti-dependency indicates a memory conflict
// that prevents safe fusion.
// Expected: No fused_kernel produced — anti-dependency blocks fusion
// ============================================================================

// CHECK-LABEL: module @test_anti_dep_blocks_fusion
// CHECK-NOT:     "ttkgir.fused_kernel"

module @test_anti_dep_blocks_fusion {
  "ttkgir.graph"() <{
    graph_name = "anti_dep_test",
    num_nodes = 2 : i32,
    num_edges = 2 : i32,
    hardware_profiles = "[]",
    dispatch_mode = "performance"
  }> ({
    "ttkgir.kernel_launch"() <{
      kernel_name = "producer",
      grid_dims = array<i64: 128, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 8192 : i32,
      register_pressure = 32 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"write", 0, true, 4194304>,
      node_id = 0 : i32
    }> : () -> ()
    "ttkgir.kernel_launch"() <{
      kernel_name = "consumer",
      grid_dims = array<i64: 128, 1, 1>,
      num_warps = 4 : i32,
      shared_memory_bytes = 8192 : i32,
      register_pressure = 32 : i32,
      memory_access_patterns = #ttkgir.mem_access_pattern<"read", 0, true, 4194304>,
      node_id = 1 : i32
    }> : () -> ()
    "ttkgir.data_dep"() <{
      source_node_id = 0 : i32,
      dest_node_id = 1 : i32,
      tensor_index = 0 : i32,
      dep_type = "flow"
    }> : () -> ()
    "ttkgir.anti_dep"() <{
      source_node_id = 0 : i32,
      dest_node_id = 1 : i32,
      tensor_index = 1 : i32,
      conflict_type = "war"
    }> : () -> ()
  }) : () -> ()
}
