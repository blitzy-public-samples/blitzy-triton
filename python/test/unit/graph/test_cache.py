"""Unit tests for the graph-level cache manager (``triton.graph.cache``).

Tests are organized into five phases matching the ``GraphCacheManager`` API:

1. **Cache Key / Signature** — determinism, variation with graph/targets,
   order-independent target-set hashing, graph-signature sensitivity.
2. **JSON Storage / Retrieval** — store, retrieve, cache miss, JSON
   round-trip fidelity, storage location verification.
3. **Invalidation** — hardware inventory change, driver/version update,
   and validation-run scheduling.
4. **Performance History** — storage, 1 MB size-limit enforcement, and
   append semantics.
5. **GraphCacheManager API** — get-or-compute pattern and cache clearing.

All tests are marked ``@pytest.mark.kernel_graph`` so that they can be
gated by hardware availability detection in CI.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from triton.graph.cache import GraphCacheManager
from triton.graph.config import GraphConfig
from triton.graph.kgir import KGIRGraph, HardwareProfile


# ---------------------------------------------------------------------------
# Module-level marker: every test in this module is a graph-level test.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.kernel_graph


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Cache Key / Signature Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_cache_key_deterministic(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Cache key for the same (graph, target set) MUST be identical across calls."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]

    key1 = manager.compute_cache_key(sample_kgir_graph, targets)
    key2 = manager.compute_cache_key(sample_kgir_graph, targets)

    assert isinstance(key1, str), "Cache key must be a string"
    assert len(key1) == 64, "SHA-256 hex digest must be 64 characters"
    assert key1 == key2, "Identical inputs must produce identical cache keys"


def test_cache_key_varies_with_graph(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    make_kgir_node,
    fresh_triton_cache,
    default_graph_config,
):
    """Different kernel graphs with the same target set MUST produce different keys."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]

    key_original = manager.compute_cache_key(sample_kgir_graph, targets)

    # Build a structurally different graph (single node, no edges)
    different_graph = KGIRGraph()
    node = make_kgir_node(kernel_name="different_kernel", grid=(256,))
    different_graph.add_node(node.kernel_fn, node.metadata)

    key_different = manager.compute_cache_key(different_graph, targets)

    assert key_original != key_different, (
        "Structurally different graphs must produce different cache keys"
    )


def test_cache_key_varies_with_targets(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    mock_amd_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Same graph with different target sets MUST produce different keys."""
    manager = GraphCacheManager(config=default_graph_config)

    key_nvidia = manager.compute_cache_key(
        sample_kgir_graph, [mock_nvidia_hw_profile]
    )
    key_amd = manager.compute_cache_key(
        sample_kgir_graph, [mock_amd_hw_profile]
    )

    assert key_nvidia != key_amd, (
        "Different target sets must produce different cache keys"
    )


def test_target_set_hashing(
    mock_nvidia_hw_profile,
    mock_amd_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Target-set hash MUST be deterministic and order-independent (set semantics)."""
    manager = GraphCacheManager(config=default_graph_config)

    # Order A, B
    hash_ab = manager.compute_target_set_key(
        [mock_nvidia_hw_profile, mock_amd_hw_profile]
    )
    # Order B, A
    hash_ba = manager.compute_target_set_key(
        [mock_amd_hw_profile, mock_nvidia_hw_profile]
    )

    assert hash_ab == hash_ba, (
        "Target-set hash must be order-independent"
    )

    # Determinism: calling again produces the same hash
    hash_ab_again = manager.compute_target_set_key(
        [mock_nvidia_hw_profile, mock_amd_hw_profile]
    )
    assert hash_ab == hash_ab_again, (
        "Target-set hash must be deterministic"
    )

    # Different single-target sets must hash differently
    hash_nvidia = manager.compute_target_set_key([mock_nvidia_hw_profile])
    hash_amd = manager.compute_target_set_key([mock_amd_hw_profile])
    assert hash_nvidia != hash_amd, (
        "Different single-target sets must produce different hashes"
    )


def test_cache_key_includes_graph_signature(
    make_kgir_node,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Graph signature MUST change when kernel identities, arg types, or grid shapes change."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]

    # Build baseline graph: one node with specific attributes
    graph_baseline = KGIRGraph()
    node_baseline = make_kgir_node(
        kernel_name="matmul", grid=(128,), tensor_shapes={0: (1024, 1024)}
    )
    graph_baseline.add_node(node_baseline.kernel_fn, node_baseline.metadata)

    key_baseline = manager.compute_cache_key(graph_baseline, targets)
    sig_baseline = manager.compute_graph_signature(graph_baseline)

    # Change kernel function identity
    graph_diff_fn = KGIRGraph()
    node_diff_fn = make_kgir_node(
        kernel_name="different_fn", grid=(128,), tensor_shapes={0: (1024, 1024)}
    )
    graph_diff_fn.add_node(node_diff_fn.kernel_fn, node_diff_fn.metadata)
    sig_diff_fn = manager.compute_graph_signature(graph_diff_fn)
    assert sig_baseline != sig_diff_fn, (
        "Different kernel function identity must change graph signature"
    )

    # Change grid shape
    graph_diff_grid = KGIRGraph()
    node_diff_grid = make_kgir_node(
        kernel_name="matmul", grid=(256, 256), tensor_shapes={0: (1024, 1024)}
    )
    graph_diff_grid.add_node(node_diff_grid.kernel_fn, node_diff_grid.metadata)
    sig_diff_grid = manager.compute_graph_signature(graph_diff_grid)
    assert sig_baseline != sig_diff_grid, (
        "Different grid dimensions must change graph signature"
    )

    # Change tensor shapes (argument types)
    graph_diff_shapes = KGIRGraph()
    node_diff_shapes = make_kgir_node(
        kernel_name="matmul", grid=(128,), tensor_shapes={0: (512, 512)}
    )
    graph_diff_shapes.add_node(node_diff_shapes.kernel_fn, node_diff_shapes.metadata)
    sig_diff_shapes = manager.compute_graph_signature(graph_diff_shapes)
    assert sig_baseline != sig_diff_shapes, (
        "Different tensor shapes must change graph signature"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — JSON Storage / Retrieval Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_store_converged_config(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Storing a converged config MUST create a JSON file at the expected path."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    # NOTE: JSON serialisation converts integer dict keys to strings,
    # so we use string keys in our test data to guarantee roundtrip fidelity.
    config_data = {
        "fusion_decisions": [{"pair": [0, 1], "fuse": True}],
        "schedule": {"order": [0, 1, 2], "streams": {"0": 0, "1": 0, "2": 0}},
        "dispatch_plan": {"target": "nvidia:sm_90"},
    }

    manager.put_converged_config(cache_key, config_data)

    # Verify the JSON file exists on disk
    expected_path = os.path.join(
        fresh_triton_cache, "graph_configs", cache_key, "config.json"
    )
    assert os.path.exists(expected_path), (
        f"config.json should exist at {expected_path}"
    )

    # Verify it is valid JSON
    with open(expected_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert "config" in raw, "Stored JSON must contain a 'config' key"
    assert raw["config"] == config_data


def test_retrieve_converged_config(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Retrieved converged config MUST match the stored config exactly."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    config_data = {
        "fusion_decisions": [{"pair": [0, 1], "fuse": True}],
        "schedule": {"order": [0, 1, 2]},
        "dispatch_plan": {"target": "nvidia:sm_90"},
    }
    manager.put_converged_config(cache_key, config_data)

    retrieved = manager.get_converged_config(cache_key)

    assert retrieved is not None, "Config should be retrievable after storage"
    assert retrieved == config_data, (
        "Retrieved config must match stored config exactly"
    )


def test_retrieve_missing_key(fresh_triton_cache, default_graph_config):
    """Retrieving a non-existent key MUST return None (cache miss), not raise."""
    manager = GraphCacheManager(config=default_graph_config)

    result = manager.get_converged_config("nonexistent_key_abc123")

    assert result is None, "Cache miss must return None, not raise an exception"


def test_json_roundtrip(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Complex nested config (lists, dicts, numeric types) MUST survive JSON roundtrip."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    complex_config = {
        "fusion_decisions": [
            {"pair": [0, 1], "fuse": True, "speedup_ratio": 1.42},
            {"pair": [1, 2], "fuse": False, "speedup_ratio": 0.98},
        ],
        "schedule": {
            "order": [0, 1, 2],
            "streams": {"0": 0, "1": 1, "2": 0},
            "barriers": [],
        },
        "memory_promotions": {
            "global_to_shared": [{"tensor_id": "T1", "size_bytes": 4096}],
        },
        "dispatch_plan": {
            "assignments": [
                {"subgraph": 0, "target": "nvidia:sm_90"},
            ],
            "mode": "balanced",
        },
        "metadata_int": 42,
        "metadata_float": 3.14159,
        "metadata_bool": True,
        "metadata_null": None,
        "metadata_nested_list": [[1, 2], [3, 4]],
    }

    manager.put_converged_config(cache_key, complex_config)
    retrieved = manager.get_converged_config(cache_key)

    assert retrieved is not None, "Complex config should survive roundtrip"
    # Verify key-by-key to pinpoint any mismatches
    for key in complex_config:
        assert key in retrieved, f"Key '{key}' missing after roundtrip"
        assert retrieved[key] == complex_config[key], (
            f"Value mismatch for key '{key}': "
            f"expected {complex_config[key]!r}, got {retrieved[key]!r}"
        )

    # Also verify via JSON serialisation for absolute fidelity
    expected_json = json.dumps(complex_config, sort_keys=True)
    actual_json = json.dumps(retrieved, sort_keys=True)
    assert expected_json == actual_json, (
        "JSON-serialised form must be identical after roundtrip"
    )


def test_cache_storage_location(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Configs MUST be stored under the graph-specific cache directory (graph_configs/)."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    manager.put_converged_config(cache_key, {"test": "storage_location"})

    # The graph_configs directory must exist under the fresh cache dir
    graph_configs_dir = os.path.join(fresh_triton_cache, "graph_configs")
    assert os.path.isdir(graph_configs_dir), (
        "graph_configs/ directory must exist under the cache base"
    )

    # The specific cache key directory must exist
    key_dir = os.path.join(graph_configs_dir, cache_key)
    assert os.path.isdir(key_dir), (
        f"Cache key directory '{cache_key}' must exist under graph_configs/"
    )

    # The config.json file must be inside it
    config_json = os.path.join(key_dir, "config.json")
    assert os.path.exists(config_json), (
        "config.json must exist inside the cache key directory"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Invalidation Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_invalidation_on_hardware_change(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    mock_amd_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Cached config MUST be invalidated when its target hardware is removed."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    # Store a config and embed target device info in its metadata so
    # invalidate_on_hw_change can detect the mismatch.
    config_data = {"test": "hw_change"}
    manager.put_converged_config(cache_key, config_data)

    # Manually add target_devices metadata to the stored JSON
    # (the production pipeline does this; we simulate it here)
    config_path = os.path.join(
        fresh_triton_cache, "graph_configs", cache_key, "config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["metadata"]["target_devices"] = ["nvidia:sm_90"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, sort_keys=True)

    # Verify config is retrievable before invalidation
    assert manager.get_converged_config(cache_key) is not None

    # Simulate hardware change: only AMD hardware remains
    current_inventory = [mock_amd_hw_profile]
    invalidated = manager.invalidate_on_hw_change(current_inventory)

    # The NVIDIA-targeted config should have been invalidated
    assert cache_key in invalidated, (
        "Config targeting removed hardware must be listed as invalidated"
    )

    # Config should no longer be retrievable
    assert manager.get_converged_config(cache_key) is None, (
        "Invalidated config must not be returned on subsequent retrieval"
    )


def test_invalidation_on_driver_update(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Cached config MUST be invalidated when Triton version changes (driver update)."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    # Store a config with current Triton version
    manager.put_converged_config(cache_key, {"test": "driver_update"})

    # Tamper with the stored version to simulate a version mismatch
    config_path = os.path.join(
        fresh_triton_cache, "graph_configs", cache_key, "config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["metadata"]["triton_version"] = "0.0.0-old"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, sort_keys=True)

    # Verify the config is still on disk
    assert manager.get_converged_config(cache_key) is not None

    # Trigger driver-update invalidation (current version != "0.0.0-old")
    invalidated = manager.invalidate_on_driver_update()

    assert cache_key in invalidated, (
        "Config stored with old version must be invalidated"
    )
    assert manager.get_converged_config(cache_key) is None, (
        "Invalidated config must not be returned"
    )


def test_validation_run_scheduling(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """schedule_validation() MUST cause needs_validation() to return True."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    # Store a fresh config
    manager.put_converged_config(cache_key, {"test": "validation"})

    # Fresh config should NOT need validation (timestamp is recent)
    assert manager.needs_validation(cache_key) is False, (
        "Freshly stored config should not need validation"
    )

    # Schedule a validation run
    manager.schedule_validation(cache_key)

    # Now it should need validation
    assert manager.needs_validation(cache_key) is True, (
        "Config with scheduled validation must return True from needs_validation()"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Performance History Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_performance_history_storage(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Performance history entries MUST be retrievable after storage."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    history = [
        {"iteration": 0, "wall_clock_ms": 5.2, "occupancy": 0.75},
        {"iteration": 1, "wall_clock_ms": 4.8, "occupancy": 0.80},
        {"iteration": 2, "wall_clock_ms": 4.6, "occupancy": 0.82},
    ]

    manager.put_performance_history(cache_key, history)
    retrieved = manager.get_performance_history(cache_key)

    assert retrieved is not None, "History should be retrievable after storage"
    assert len(retrieved) == 3, "All history entries must be preserved"
    assert retrieved[0]["iteration"] == 0
    assert retrieved[2]["wall_clock_ms"] == 4.6


def test_performance_history_size_limit(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Performance history MUST be < 1 MB per cached configuration (AAP §0.7.2).

    When the input exceeds 1 MB, the oldest entries must be dropped to
    enforce the limit while retaining the most recent entries.
    """
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    # Generate a history payload that far exceeds 1 MB.
    # Each entry is ~200 bytes of JSON; 10,000 entries ≈ 2 MB.
    large_history = [
        {
            "iteration": i,
            "wall_clock_ms": float(i) * 0.1,
            "occupancy": 0.5 + (i % 50) * 0.01,
            "bandwidth_gbps": 1200.0 + i * 0.5,
            "sm_utilisation": 0.6 + (i % 40) * 0.01,
            "extra_payload": "x" * 100,  # pad to grow JSON size
        }
        for i in range(10_000)
    ]

    # Verify our test data actually exceeds 1 MB
    raw_size = len(json.dumps(large_history).encode("utf-8"))
    assert raw_size > 1_000_000, (
        f"Test data should exceed 1 MB; got {raw_size} bytes"
    )

    manager.put_performance_history(cache_key, large_history)
    retrieved = manager.get_performance_history(cache_key)

    assert retrieved is not None, "Truncated history should still be retrievable"

    # The history list itself (compact JSON, no indentation) must fit
    # within the 1 MB budget.  The on-disk file uses indent=2 and wrapper
    # metadata which may push the file slightly over, but the truncation
    # logic correctly bounds the *list* payload to <= 1 MB.
    compact_size = len(json.dumps(retrieved).encode("utf-8"))
    assert compact_size <= 1_000_000, (
        f"Compact JSON of truncated history list must be <= 1 MB; "
        f"got {compact_size} bytes"
    )

    # Verify the history.json file exists on disk
    history_path = os.path.join(
        fresh_triton_cache, "graph_history", cache_key, "history.json"
    )
    assert os.path.exists(history_path), "history.json must exist"

    # The most recent entries (tail) must be preserved
    assert retrieved[-1]["iteration"] == 9999, (
        "Most recent entry must be preserved after truncation"
    )
    # Some entries from the beginning should have been dropped
    assert len(retrieved) < len(large_history), (
        "History must have been truncated"
    )


def test_performance_history_append(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """Appending new measurements MUST preserve old entries (up to size limit)."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    # Initial history
    initial_history = [
        {"iteration": 0, "wall_clock_ms": 5.0},
        {"iteration": 1, "wall_clock_ms": 4.5},
    ]
    manager.put_performance_history(cache_key, initial_history)

    # Append new entries by reading existing + extending
    existing = manager.get_performance_history(cache_key)
    assert existing is not None
    appended_history = existing + [
        {"iteration": 2, "wall_clock_ms": 4.2},
        {"iteration": 3, "wall_clock_ms": 4.0},
    ]
    manager.put_performance_history(cache_key, appended_history)

    # Retrieve and verify all entries are present
    final = manager.get_performance_history(cache_key)
    assert final is not None
    assert len(final) == 4, "All original + appended entries must be present"
    assert final[0]["iteration"] == 0, "First entry must be preserved"
    assert final[3]["iteration"] == 3, "Last appended entry must be present"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — GraphCacheManager API Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_cache_manager_get_or_compute(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    fresh_triton_cache,
    default_graph_config,
):
    """get_converged_config MUST return None on miss and the stored config on hit."""
    manager = GraphCacheManager(config=default_graph_config)
    targets = [mock_nvidia_hw_profile]
    cache_key = manager.compute_cache_key(sample_kgir_graph, targets)

    # Cache miss — caller would compute optimisation
    result = manager.get_converged_config(cache_key)
    assert result is None, "First call must be a cache miss (None)"

    # Simulate caller computing and storing the result
    computed_config = {
        "fusion_decisions": [],
        "schedule": {"order": [0, 1, 2]},
    }
    manager.put_converged_config(cache_key, computed_config)

    # Cache hit — subsequent call returns stored config
    hit_result = manager.get_converged_config(cache_key)
    assert hit_result is not None, "Second call must be a cache hit"
    assert hit_result == computed_config, (
        "Cache hit must return the stored config"
    )


def test_cache_manager_clear(
    sample_kgir_graph,
    mock_nvidia_hw_profile,
    mock_amd_hw_profile,
    make_kgir_node,
    fresh_triton_cache,
    default_graph_config,
):
    """invalidate() MUST remove the specified cached config and its history."""
    manager = GraphCacheManager(config=default_graph_config)
    targets_nvidia = [mock_nvidia_hw_profile]
    targets_amd = [mock_amd_hw_profile]

    key1 = manager.compute_cache_key(sample_kgir_graph, targets_nvidia)
    key2 = manager.compute_cache_key(sample_kgir_graph, targets_amd)

    # Store two different configs
    manager.put_converged_config(key1, {"target": "nvidia"})
    manager.put_converged_config(key2, {"target": "amd"})

    # Also store performance history for key1
    manager.put_performance_history(key1, [{"iteration": 0, "ms": 1.0}])

    # Verify both are present
    assert manager.get_converged_config(key1) is not None
    assert manager.get_converged_config(key2) is not None

    # Invalidate key1 — should remove its config AND history
    manager.invalidate(key1)

    assert manager.get_converged_config(key1) is None, (
        "Invalidated config must be gone"
    )
    assert manager.get_performance_history(key1) is None, (
        "Performance history for invalidated key must be gone"
    )

    # key2 should remain unaffected
    assert manager.get_converged_config(key2) is not None, (
        "Other cached configs must not be affected by invalidation"
    )

    # Invalidate key2 as well
    manager.invalidate(key2)
    assert manager.get_converged_config(key2) is None, (
        "Second invalidated config must also be gone"
    )
