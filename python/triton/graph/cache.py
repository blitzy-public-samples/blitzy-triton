"""Graph-level cache manager for converged optimization configurations.

Implements caching for the graph-level kernel optimization layer. Cache
entries are keyed by ``(kernel graph signature, hardware target set)`` tuples.
Follows the existing ``FileCacheManager`` pattern from ``triton.runtime.cache``
using JSON serialization with atomic writes via temp directories.

Cache directory structure under ``~/.triton/cache/``::

    graph_configs/      Converged configs per (graph_sig, target_set)
    graph_calibration/  Per-target calibration microbenchmark data
    graph_history/      Per-config execution time history for regression detection
    hw_profiles/        Per-device hardware profile descriptors
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional

from triton.graph.config import GraphConfig
from triton import knobs


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum size for performance history log per cached configuration (1 MB).
# Per AAP §0.7.2: Performance history log MUST be < 1MB per cached configuration.
# Uses the strict SI definition of 1 MB = 1,000,000 bytes.
_MAX_HISTORY_SIZE_BYTES: int = 1_000_000

# Default age threshold (seconds) before a cached config needs revalidation.
# Set to 24 hours as a sensible production default.
_DEFAULT_VALIDATION_AGE_THRESHOLD_SECONDS: int = 86400

# Subdirectory names for the graph-level cache hierarchy.
# These live under the Triton cache base directory (typically ~/.triton/cache/).
_GRAPH_CONFIGS_DIR: str = "graph_configs"
_GRAPH_CALIBRATION_DIR: str = "graph_calibration"
_GRAPH_HISTORY_DIR: str = "graph_history"
_HW_PROFILES_DIR: str = "hw_profiles"


class GraphCacheManager:
    """Graph-level cache for converged optimization configurations.

    Provides storage and retrieval for:

    * **Converged configurations** — fusion decisions, scheduling plans,
      dispatch assignments, and memory promotions keyed by (graph signature,
      target set).
    * **Calibration data** — per-target cost-model launch-overhead
      microbenchmark results and heuristic parameters.
    * **Performance history** — per-configuration execution-time history
      for regression detection (enforced < 1 MB per entry).
    * **Hardware profiles** — per-device SM count, SMEM capacity, register
      file size, memory bandwidth, and other characteristics.

    Concurrency Safety
    ------------------
    All write operations use an atomic temp-directory-then-replace pattern
    (matching ``FileCacheManager`` behaviour) to prevent corrupted reads
    during concurrent access from multiple processes.

    Example
    -------
    ::

        manager = GraphCacheManager(config=GraphConfig())
        key = manager.compute_cache_key(graph, targets)
        cached = manager.get_converged_config(key)
        if cached is None:
            # Run optimization pipeline …
            manager.put_converged_config(key, optimized_config)
    """

    def __init__(self, config: Optional[GraphConfig] = None) -> None:
        """Initialise the GraphCacheManager.

        Resolves the cache base directory via ``triton.knobs.cache.dir``
        (``TRITON_CACHE_DIR`` environment variable), falling back to
        ``triton.knobs.cache.get_triton_dir("cache")`` which resolves to
        ``~/.triton/cache/``.

        Args:
            config: Optional ``GraphConfig`` providing graph-level settings.
                Attributes ``dump_kgir`` and ``max_graph_nodes`` are read
                from the config when provided.  If ``None``, default values
                are used.
        """
        self._config = config

        # Resolve cache base directory following the established
        # FileCacheManager pattern from triton.runtime.cache.
        # Primary: knobs.cache.dir (TRITON_CACHE_DIR) which defaults to
        # knobs.cache.get_triton_dir("cache") → ~/.triton/cache/
        cache_base: str = knobs.cache.dir
        if not cache_base:
            # Explicit fallback via get_triton_dir for robustness
            cache_base = knobs.cache.get_triton_dir("cache")

        self._cache_dir: str = cache_base

        # Pre-compute subdirectory paths for each cache category
        self._graph_configs_dir: str = os.path.join(
            self._cache_dir, _GRAPH_CONFIGS_DIR
        )
        self._graph_calibration_dir: str = os.path.join(
            self._cache_dir, _GRAPH_CALIBRATION_DIR
        )
        self._graph_history_dir: str = os.path.join(
            self._cache_dir, _GRAPH_HISTORY_DIR
        )
        self._hw_profiles_dir: str = os.path.join(
            self._cache_dir, _HW_PROFILES_DIR
        )

        # Extract configuration values (accessed members per schema contract)
        if config is not None:
            self._dump_kgir: bool = config.dump_kgir
            self._max_graph_nodes: int = config.max_graph_nodes
        else:
            self._dump_kgir = False
            self._max_graph_nodes = 100

    # ------------------------------------------------------------------
    # Cache Key Computation
    # ------------------------------------------------------------------

    def compute_graph_signature(self, graph: Any) -> str:
        """Compute a unique SHA-256 signature for a kernel graph structure.

        The hash incorporates (in topological order):

        * Kernel function cache keys (or name-based hashes as fallback)
        * Tensor argument shapes and dtypes per kernel node
        * Launch grid dimensions per kernel node
        * Dependency edges (source → target, edge type)

        This guarantees that structurally identical graphs produce identical
        signatures while any structural change produces a different one.

        Args:
            graph: A ``KGIRGraph`` instance (or any object exposing
                ``topological_sort()``, ``get_node(node_id)``, and
                ``get_edges()``).

        Returns:
            A hexadecimal SHA-256 digest uniquely identifying the graph.
        """
        h = hashlib.sha256()

        # Obtain deterministic node ordering via topological sort.
        # Fall back to sorted node IDs if topological_sort() is unavailable.
        try:
            topo_order = graph.topological_sort()
        except Exception:
            nodes_dict = getattr(graph, "_nodes", {})
            topo_order = sorted(nodes_dict.keys())

        for node_id in topo_order:
            try:
                node = graph.get_node(node_id)
            except Exception:
                continue

            # Hash the node identifier for positional integrity
            h.update(f"node:{node_id}".encode("utf-8"))

            # Hash kernel function identity (cache key preferred)
            kernel_fn = getattr(node, "kernel_fn", None)
            if kernel_fn is not None:
                cache_key = getattr(kernel_fn, "cache_key", None)
                if cache_key is not None:
                    h.update(f"kfn:{cache_key}".encode("utf-8"))
                else:
                    fn_name = getattr(
                        kernel_fn, "__name__", str(id(kernel_fn))
                    )
                    h.update(f"kfn:{fn_name}".encode("utf-8"))

            # Hash node metadata — shapes, dtypes, grid dimensions
            metadata = getattr(node, "metadata", None)
            if metadata is not None:
                # Tensor shapes (sorted by argument index for determinism)
                tensor_shapes = getattr(metadata, "tensor_shapes", {})
                for arg_idx in sorted(tensor_shapes.keys()):
                    h.update(
                        f"shape:{arg_idx}:{tensor_shapes[arg_idx]}".encode(
                            "utf-8"
                        )
                    )

                # Tensor dtypes (sorted by argument index)
                tensor_dtypes = getattr(metadata, "tensor_dtypes", {})
                for arg_idx in sorted(tensor_dtypes.keys()):
                    h.update(
                        f"dtype:{arg_idx}:{tensor_dtypes[arg_idx]}".encode(
                            "utf-8"
                        )
                    )

                # Grid dimensions
                grid_dims = getattr(metadata, "grid_dimensions", ())
                h.update(f"grid:{grid_dims}".encode("utf-8"))

        # Hash dependency edges for structural completeness
        try:
            edges = graph.get_edges()
        except Exception:
            edges = getattr(graph, "_edges", [])

        for edge in edges:
            src = getattr(edge, "source_id", "")
            tgt = getattr(edge, "target_id", "")
            etype = getattr(edge, "edge_type", "")
            h.update(f"edge:{src}->{tgt}:{etype}".encode("utf-8"))

        return h.hexdigest()

    def compute_target_set_key(self, targets: List[Any]) -> str:
        """Compute an order-independent SHA-256 hash for a set of GPU targets.

        Each target's ``(backend, arch, warp_size)`` descriptor is sorted
        lexicographically before hashing, ensuring identical target sets
        produce identical keys regardless of input order.

        Args:
            targets: A list of ``GPUTarget``-like objects (or dicts) with
                ``backend``, ``arch``, and ``warp_size`` attributes/keys.

        Returns:
            A hexadecimal SHA-256 digest identifying this target set.
        """
        h = hashlib.sha256()

        target_descriptors: List[str] = []
        for target in targets:
            if isinstance(target, dict):
                backend = str(target.get("backend", ""))
                arch = str(target.get("arch", ""))
                warp_size = str(target.get("warp_size", ""))
            else:
                backend = str(getattr(target, "backend", ""))
                arch = str(getattr(target, "arch", ""))
                warp_size = str(getattr(target, "warp_size", ""))
            target_descriptors.append(f"{backend}:{arch}:{warp_size}")

        # Sort for order-independence
        target_descriptors.sort()

        for desc in target_descriptors:
            h.update(desc.encode("utf-8"))

        return h.hexdigest()

    def compute_cache_key(self, graph: Any, targets: List[Any]) -> str:
        """Compute the primary cache key combining graph signature and target set.

        This is the main lookup key for converged optimization configurations.
        It is formed by hashing the concatenation of the graph signature and
        target set key, producing a single composite key.

        Args:
            graph: A ``KGIRGraph`` instance.
            targets: A list of ``GPUTarget``-like objects.

        Returns:
            A hexadecimal SHA-256 digest for the (graph, target set) pair.
        """
        graph_sig = self.compute_graph_signature(graph)
        target_key = self.compute_target_set_key(targets)

        combined = hashlib.sha256()
        combined.update(graph_sig.encode("utf-8"))
        combined.update(target_key.encode("utf-8"))

        return combined.hexdigest()

    # ------------------------------------------------------------------
    # Converged Configuration Operations
    # ------------------------------------------------------------------

    def get_converged_config(self, cache_key: str) -> Optional[Dict]:
        """Look up a converged optimization configuration by cache key.

        Reads from ``graph_configs/{cache_key}/config.json``.  The stored
        configuration typically includes fusion decisions, scheduling plans,
        dispatch assignments, and memory promotion decisions.

        Args:
            cache_key: The cache key from ``compute_cache_key()``.

        Returns:
            The cached configuration dict (the ``"config"`` sub-object),
            or ``None`` if the key is not cached.
        """
        config_path = os.path.join(
            self._graph_configs_dir, cache_key, "config.json"
        )
        raw = self._read_json(config_path)
        if raw is None:
            return None
        # Stored format wraps user data in {"config": ..., "metadata": ...}
        if isinstance(raw, dict) and "config" in raw:
            return raw["config"]
        return raw

    def put_converged_config(self, cache_key: str, config: Dict) -> None:
        """Store a converged optimization configuration as JSON.

        Writes to ``graph_configs/{cache_key}/config.json`` with metadata
        including: timestamp, Triton version (for invalidation), and the
        ``validation_scheduled`` flag (initially ``False``).

        Uses atomic write (temp directory → ``os.replace``) following the
        ``FileCacheManager`` pattern for concurrency safety.

        Args:
            cache_key: The cache key from ``compute_cache_key()``.
            config: The configuration dict to store (fusion decisions,
                scheduling plan, dispatch assignments, etc.).
        """
        from triton import __version__

        data: Dict[str, Any] = {
            "config": config,
            "metadata": {
                "timestamp": time.time(),
                "triton_version": __version__,
                "validation_scheduled": False,
            },
        }

        target_dir = os.path.join(self._graph_configs_dir, cache_key)
        self._write_json_atomic(target_dir, "config.json", data)

    # ------------------------------------------------------------------
    # Calibration Data Operations
    # ------------------------------------------------------------------

    def get_calibration_data(self, target_key: str) -> Optional[Dict]:
        """Get per-target cost model calibration data.

        Reads from ``graph_calibration/{target_key}/calibration.json``.
        Contains launch overhead microbenchmark results and heuristic
        parameters used by the adaptive cost model.

        Args:
            target_key: A target-specific key (e.g. from
                ``compute_target_set_key`` with a single target).

        Returns:
            The calibration data dict, or ``None`` if not cached.
        """
        path = os.path.join(
            self._graph_calibration_dir, target_key, "calibration.json"
        )
        return self._read_json(path)

    def put_calibration_data(self, target_key: str, data: Dict) -> None:
        """Store per-target cost model calibration data as JSON.

        Writes to ``graph_calibration/{target_key}/calibration.json``.

        Args:
            target_key: A target-specific key.
            data: The calibration data dict (launch overhead, bandwidth
                parameters, heuristic weights, etc.).
        """
        target_dir = os.path.join(self._graph_calibration_dir, target_key)
        self._write_json_atomic(target_dir, "calibration.json", data)

    # ------------------------------------------------------------------
    # Performance History Operations
    # ------------------------------------------------------------------

    def get_performance_history(
        self, cache_key: str
    ) -> Optional[List[Dict]]:
        """Get execution time history for a cached configuration.

        Reads from ``graph_history/{cache_key}/history.json``.  The history
        is stored as a list of dicts, each representing one execution run
        with timing metrics.

        Performance history is kept under 1 MB per cached configuration
        as required by AAP §0.7.2.

        Args:
            cache_key: The cache key from ``compute_cache_key()``.

        Returns:
            A list of history entry dicts, or ``None`` if no history.
        """
        path = os.path.join(
            self._graph_history_dir, cache_key, "history.json"
        )
        raw = self._read_json(path)
        if raw is None:
            return None
        # Stored format: {"history": [...], "last_updated": ..., ...}
        if isinstance(raw, dict) and "history" in raw:
            return raw["history"]
        if isinstance(raw, list):
            return raw
        return None

    def put_performance_history(
        self, cache_key: str, history: List[Dict]
    ) -> None:
        """Store performance history as JSON, enforcing the 1 MB limit.

        Writes to ``graph_history/{cache_key}/history.json``.  If the
        serialised history exceeds ``_MAX_HISTORY_SIZE_BYTES`` (1 MB),
        the oldest entries are dropped to fit within the budget while
        retaining the most recent entries.

        Args:
            cache_key: The cache key.
            history: A list of history entry dicts to persist.
        """
        truncated = self._truncate_history_to_size_limit(history)

        data: Dict[str, Any] = {
            "history": truncated,
            "last_updated": time.time(),
            "entry_count": len(truncated),
        }

        target_dir = os.path.join(self._graph_history_dir, cache_key)
        self._write_json_atomic(target_dir, "history.json", data)

    # ------------------------------------------------------------------
    # Hardware Profile Operations
    # ------------------------------------------------------------------

    def get_hardware_profile(self, device_key: str) -> Optional[Dict]:
        """Get a cached hardware profile descriptor.

        Reads from ``hw_profiles/{device_key}/profile.json``.  Contains
        per-device SM count, SMEM capacity, register file size, memory
        bandwidth, compute throughput, and other characteristics.

        Args:
            device_key: A device-specific key identifying the hardware
                (e.g. ``"nvidia:sm_90"``).

        Returns:
            The hardware profile dict, or ``None`` if not cached.
        """
        path = os.path.join(
            self._hw_profiles_dir, device_key, "profile.json"
        )
        return self._read_json(path)

    def put_hardware_profile(self, device_key: str, profile: Dict) -> None:
        """Store a hardware profile descriptor as JSON.

        Writes to ``hw_profiles/{device_key}/profile.json``.

        Args:
            device_key: A device-specific key.
            profile: The hardware profile dict matching the
                ``HardwareProfile`` schema.
        """
        target_dir = os.path.join(self._hw_profiles_dir, device_key)
        self._write_json_atomic(target_dir, "profile.json", profile)

    # ------------------------------------------------------------------
    # Cache Invalidation
    # ------------------------------------------------------------------

    def invalidate(self, cache_key: str) -> None:
        """Remove a cached configuration and all associated data.

        Cleans up the converged configuration directory and the
        performance history directory for the given cache key.

        Args:
            cache_key: The cache key to invalidate.
        """
        # Remove converged configuration
        config_dir = os.path.join(self._graph_configs_dir, cache_key)
        self._remove_dir_safe(config_dir)

        # Remove associated performance history
        history_dir = os.path.join(self._graph_history_dir, cache_key)
        self._remove_dir_safe(history_dir)

    def invalidate_on_hw_change(
        self, current_inventory: List[Any]
    ) -> List[str]:
        """Invalidate configs whose target set no longer matches hardware.

        Scans all cached configurations and compares their recorded target
        device keys against the current hardware inventory.  Any config
        referencing hardware that is no longer present is invalidated.

        Args:
            current_inventory: A list of ``HardwareProfile``-like objects
                (or dicts) describing the currently available hardware.
                Expected attributes/keys: ``vendor``, ``arch_generation``.

        Returns:
            A list of cache keys that were invalidated.
        """
        invalidated_keys: List[str] = []

        # Build set of current device keys from inventory
        current_device_keys: set = set()
        for device in current_inventory:
            if isinstance(device, dict):
                vendor = device.get("vendor", "")
                arch = device.get("arch_generation", "")
            else:
                vendor = getattr(device, "vendor", "")
                arch = getattr(device, "arch_generation", "")
            current_device_keys.add(f"{vendor}:{arch}")

        if not os.path.isdir(self._graph_configs_dir):
            return invalidated_keys

        try:
            entries = os.listdir(self._graph_configs_dir)
        except OSError:
            return invalidated_keys

        for cache_key in entries:
            entry_path = os.path.join(self._graph_configs_dir, cache_key)
            if not os.path.isdir(entry_path):
                continue

            config_path = os.path.join(entry_path, "config.json")
            raw = self._read_json(config_path)
            if raw is None:
                continue

            metadata = (
                raw.get("metadata", {}) if isinstance(raw, dict) else {}
            )
            target_devices = metadata.get("target_devices", [])

            if target_devices:
                # If any target device is no longer in the inventory,
                # the cached configuration is stale and must be removed.
                for dev_key in target_devices:
                    if dev_key not in current_device_keys:
                        self.invalidate(cache_key)
                        invalidated_keys.append(cache_key)
                        break

        return invalidated_keys

    def invalidate_on_driver_update(self) -> List[str]:
        """Invalidate configs stored with a different Triton version.

        Scans all cached configurations and compares the stored
        ``triton_version`` metadata against the current runtime version.
        Configs with a version mismatch are invalidated because driver /
        compiler changes may produce different compilation artefacts.

        Returns:
            A list of cache keys that were invalidated.
        """
        from triton import __version__

        invalidated_keys: List[str] = []

        if not os.path.isdir(self._graph_configs_dir):
            return invalidated_keys

        try:
            entries = os.listdir(self._graph_configs_dir)
        except OSError:
            return invalidated_keys

        for cache_key in entries:
            entry_path = os.path.join(self._graph_configs_dir, cache_key)
            if not os.path.isdir(entry_path):
                continue

            config_path = os.path.join(entry_path, "config.json")
            raw = self._read_json(config_path)
            if raw is None:
                continue

            metadata = (
                raw.get("metadata", {}) if isinstance(raw, dict) else {}
            )
            stored_version = metadata.get("triton_version", "")

            if stored_version and stored_version != __version__:
                self.invalidate(cache_key)
                invalidated_keys.append(cache_key)

        return invalidated_keys

    # ------------------------------------------------------------------
    # Validation Run Scheduling
    # ------------------------------------------------------------------

    def needs_validation(self, cache_key: str) -> bool:
        """Check whether a cached configuration needs a validation run.

        A validation run re-executes the cached configuration and checks
        for performance regressions.  Validation is needed when:

        1. The ``validation_scheduled`` flag is set (via
           ``schedule_validation``).
        2. The configuration is older than the age threshold (24 h).

        Args:
            cache_key: The cache key to check.

        Returns:
            ``True`` if a validation run should be performed.
        """
        config_path = os.path.join(
            self._graph_configs_dir, cache_key, "config.json"
        )
        raw = self._read_json(config_path)
        if raw is None:
            # No cached configuration — nothing to validate
            return False

        metadata = (
            raw.get("metadata", {}) if isinstance(raw, dict) else {}
        )

        # Explicitly scheduled validation takes priority
        if metadata.get("validation_scheduled", False):
            return True

        # Age-based validation: stale configs need revalidation
        stored_ts = metadata.get("timestamp", 0)
        if stored_ts > 0:
            age = time.time() - stored_ts
            if age > _DEFAULT_VALIDATION_AGE_THRESHOLD_SECONDS:
                return True

        return False

    def schedule_validation(self, cache_key: str) -> None:
        """Mark a cached configuration for validation on next execution.

        Sets the ``validation_scheduled`` flag in the configuration
        metadata.  The next call to ``needs_validation`` will return
        ``True`` for this key.

        After a successful validation run, the caller should clear the
        flag by writing an updated configuration via
        ``put_converged_config``.

        Args:
            cache_key: The cache key to schedule validation for.
        """
        config_path = os.path.join(
            self._graph_configs_dir, cache_key, "config.json"
        )
        raw = self._read_json(config_path)
        if raw is None:
            return
        if not isinstance(raw, dict):
            return

        metadata = raw.get("metadata", {})
        metadata["validation_scheduled"] = True
        metadata["validation_scheduled_at"] = time.time()
        raw["metadata"] = metadata

        target_dir = os.path.join(self._graph_configs_dir, cache_key)
        self._write_json_atomic(target_dir, "config.json", raw)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _read_json(self, path: str) -> Optional[Any]:
        """Read and parse a JSON file.

        Returns ``None`` if the file does not exist, is unreadable, or
        contains malformed JSON.

        Args:
            path: Absolute filesystem path to the JSON file.

        Returns:
            Parsed JSON object, or ``None`` on any failure.
        """
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return None

    @staticmethod
    def _deserialise_json_string(data_str: str) -> Optional[Any]:
        """Parse a JSON string into a Python object.

        Utility for deserialising JSON from in-memory strings (used for
        size estimation and data interchange within the cache layer).

        Args:
            data_str: A JSON-encoded string.

        Returns:
            Parsed Python object, or ``None`` on decode failure.
        """
        try:
            return json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return None

    def _write_json_atomic(
        self, target_dir: str, filename: str, data: Any
    ) -> None:
        """Atomically write JSON data following the FileCacheManager pattern.

        Procedure:

        1. Ensure ``target_dir`` exists.
        2. Create a temp directory ``{target_dir}/.tmp-{uuid4}-{pid}``.
        3. Write the JSON file inside the temp directory.
        4. Atomically move (``os.replace``) to the final location.
        5. Clean up the temp directory (``os.removedirs``).

        This prevents partially-written files from being read by
        concurrent processes.

        Args:
            target_dir: Directory where the file should live.
            filename: Name of the JSON file (e.g. ``config.json``).
            data: Data to serialise as JSON.
        """
        os.makedirs(target_dir, exist_ok=True)

        # Create a temp directory with a unique name (uuid4 + pid)
        # following the exact FileCacheManager pattern.
        temp_id = f"{uuid.uuid4()}-{os.getpid()}"
        temp_dir = os.path.join(target_dir, f".tmp-{temp_id}")

        try:
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, filename)
            target_file = os.path.join(target_dir, filename)

            # Write JSON to temp file
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)

            # Atomic replace (POSIX guarantees atomicity for os.replace)
            os.replace(temp_file, target_file)

            # Clean up the now-empty temp directory.
            # os.removedirs removes the directory and tries to remove
            # empty parent directories, matching FileCacheManager behaviour.
            try:
                os.removedirs(temp_dir)
            except OSError:
                pass

        except OSError:
            # Cache writes are best-effort; silently handle failures
            # but ensure temp directory is always cleaned up.
            try:
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass

    def _remove_dir_safe(self, dir_path: str) -> None:
        """Safely remove a directory tree.

        Uses ``shutil.rmtree`` with error suppression for robustness.
        No-op if the directory does not exist.

        Args:
            dir_path: Directory to remove.
        """
        if os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path, ignore_errors=True)
            except OSError:
                pass

    @staticmethod
    def _truncate_history_to_size_limit(
        history: List[Dict],
    ) -> List[Dict]:
        """Truncate performance history to stay within the 1 MB limit.

        Keeps the most recent entries (end of list) and drops the oldest
        until the serialised JSON fits within ``_MAX_HISTORY_SIZE_BYTES``.
        Uses a binary search to efficiently find the maximum number of
        recent entries that fit.

        Args:
            history: The full history list (chronological order,
                oldest first).

        Returns:
            A (possibly truncated) list fitting within 1 MB.
        """
        if not history:
            return history

        # Quick check: does the full history already fit?
        serialised = json.dumps(history)
        if len(serialised.encode("utf-8")) <= _MAX_HISTORY_SIZE_BYTES:
            return history

        # Binary search for the maximum number of tail entries that fit.
        lo, hi = 0, len(history)
        best = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            if mid == 0:
                best = 0
                break
            subset = history[-mid:]
            subset_size = len(json.dumps(subset).encode("utf-8"))
            if subset_size <= _MAX_HISTORY_SIZE_BYTES:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return history[-best:] if best > 0 else []
