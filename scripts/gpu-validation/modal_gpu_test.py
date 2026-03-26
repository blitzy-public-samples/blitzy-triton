"""
Modal-based GPU validation script for Triton KGIR graph-level optimization layer.

Provisions 2-GPU containers (A100 or H100) on Modal's serverless infrastructure
and runs the Triton KGIR test suite in 3 phases:
  - Phase 1: Single-GPU tests (not multi_device, not heterogeneous_hw)
  - Phase 2: Multi-GPU same-generation tests (multi_device, not heterogeneous_hw)
  - Phase 3: Heterogeneous hardware tests (heterogeneous_hw)

Usage:
  modal run scripts/gpu-validation/modal_gpu_test.py --gpu-type A100 --phase all
  modal run scripts/gpu-validation/modal_gpu_test.py --gpu-type H100 --phase phase1
"""

from __future__ import annotations

import base64
import os
import sys
import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/blitzy-public-samples/blitzy-triton.git"
BRANCH = "blitzy-cf40add8-bcfd-4a9e-8be7-5a8f10b85bb4"
RESULTS_DIR = "/tmp/results"
TRITON_SRC = "/root/triton"

# Environment variables set inside the GPU container for KGIR tests.
# PYTHONPATH ensures our editable-install triton package is found BEFORE any
# stale triton artifacts that PyTorch's conda base image may leave behind.
# Without this, Python can resolve ``triton`` as a namespace package
# (``__file__`` = None) which causes ``from triton import __version__`` to fail.
KGIR_ENV = {
    "TRITON_FUSION_THRESHOLD": "0.10",
    "TRITON_FEEDBACK_ENABLE": "1",
    "TRITON_DISPATCH_MODE": "performance",
    "TRITON_KGIR_DUMP": "1",
    "PYTHONPATH": f"{TRITON_SRC}/python",
}

# Phase definitions: (phase_name, pytest marker expression)
PHASES = {
    "phase1": ("phase1", "not multi_device and not heterogeneous_hw"),
    "phase2": ("phase2", "multi_device and not heterogeneous_hw"),
    "phase3": ("phase3", "heterogeneous_hw"),
}

# Timeout for the entire remote function (50 minutes to allow build + tests).
REMOTE_TIMEOUT = 3000

# ---------------------------------------------------------------------------
# Modal Image — built once, cached across runs
# ---------------------------------------------------------------------------
triton_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel",
    )
    .apt_install("git", "clang", "lld")
    .pip_install(
        "cmake>=3.20",
        "ninja>=1.11.1",
        "pybind11>=2.13.1",
        "setuptools>=40.8.0",
        "wheel",
        "lit",
        "pytest",
        "pytest-xdist",
        "pytest-forked",
        "scipy>=1.7.1",
        "numpy",
        "autopep8",
        "isort",
    )
    .run_commands(
        f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {TRITON_SRC}",
    )
    .run_commands(
        # ---------------------------------------------------------------
        # NUCLEAR CLEANUP of pre-installed triton from the PyTorch base
        # image.  PyTorch 2.4.0-cuda12.4 ships triton ~3.0 via conda.
        # ``pip uninstall`` alone is insufficient because the conda package
        # leaves artefacts (empty ``triton/`` dirs, ``.dist-info/``,
        # namespace package stubs) that cause Python to resolve ``triton``
        # as a *namespace* package (``__file__`` = None, ``__version__``
        # missing), which then triggers:
        #   ImportError: cannot import name '__version__' from 'triton'
        #               (unknown location)
        # and cascading "partially initialized module" errors.
        #
        # The fix is a three-layer removal: pip → conda → filesystem nuke,
        # followed by verification that ``import triton`` actually FAILS.
        # ---------------------------------------------------------------
        # Layer 1: pip uninstall
        "pip uninstall -y triton pytorch-triton triton-nightly triton-key 2>/dev/null; "
        # Layer 2: conda remove (force, don't resolve deps)
        "conda remove -y --force triton pytorch-triton 2>/dev/null; "
        # Layer 3: filesystem nuke — remove ALL triton-related paths from
        # every site-packages directory in the conda environment
        "for SITE in $(python -c \"import site; print(' '.join(site.getsitepackages()))\"); do "
        "  echo \"Cleaning $SITE ...\"; "
        "  rm -rf $SITE/triton $SITE/triton-* $SITE/triton_* "
        "         $SITE/_triton* $SITE/pytorch_triton* "
        "         $SITE/torch/_triton 2>/dev/null; "
        "  find $SITE -maxdepth 1 -name '*triton*.dist-info' -exec rm -rf {} + 2>/dev/null; "
        "  find $SITE -maxdepth 1 -name '*triton*.egg-info' -exec rm -rf {} + 2>/dev/null; "
        "  find $SITE -maxdepth 1 -name '*triton*.egg-link' -delete 2>/dev/null; "
        "  find $SITE -maxdepth 1 -name '*triton*.pth' -delete 2>/dev/null; "
        "  for pth in $SITE/*.pth; do "
        "    grep -qi triton \"$pth\" 2>/dev/null && echo \"  Removing .pth referencing triton: $pth\" && rm -f \"$pth\"; "
        "  done; "
        "done; "
        # Verify triton is completely gone (use find_spec to avoid import)
        "python -c \""
        "import importlib.util, sys; "
        "spec = importlib.util.find_spec('triton'); "
        "print('FAIL: triton still findable at ' + str(getattr(spec, 'origin', '?')) if spec else 'OK: triton fully removed — clean slate for editable install'); "
        "sys.exit(1 if spec else 0)"
        "\" && "
        "echo 'Pre-existing triton packages removed successfully.'",
    )
    .run_commands(
        f"cd {TRITON_SRC} && "
        "TRITON_BUILD_PROTON=OFF TRITON_BUILD_WITH_O1=true MAX_JOBS=4 "
        "pip install -v -e '.[tests]' --no-build-isolation",
    )
    .run_commands(
        # Fix GLIBCXX_3.4.30: libtriton.so requires newer libstdc++ symbols than
        # the conda-bundled version provides.  The CUDA devel base image ships a
        # system libstdc++ that has the required symbols — copy it over conda's
        # older version so that torch/triton find the right one at runtime.
        "cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6 && "
        "ldconfig && "
        "echo 'GLIBCXX fix applied — verifying:' && "
        "strings /opt/conda/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30",
    )
    .run_commands(
        # Verify triton is correctly importable after build — test both
        # with and without PYTHONPATH to confirm the editable install
        # works AND the PYTHONPATH override works.
        "echo '=== Build verification (default sys.path) ===' && "
        "python -c \""
        "import triton; "
        "print(f'triton.__version__ = {triton.__version__}'); "
        "print(f'triton.__file__   = {triton.__file__}'); "
        "print(f'triton.__path__   = {list(triton.__path__)}'); "
        "import triton.graph; "
        "print('triton.graph imported OK'); "
        "import triton.backends; "
        "print(f'backends discovered: {list(triton.backends.backends.keys()) if hasattr(triton.backends, \\\"backends\\\") else \\\"NONE\\\"}'); "
        "\" && "
        "echo '=== Build verification (PYTHONPATH override) ===' && "
        f"PYTHONPATH={TRITON_SRC}/python python -c \""
        "import sys; "
        "print(f'sys.path[0:3] = {sys.path[0:3]}'); "
        "import triton; "
        "assert hasattr(triton, '__version__'), 'triton.__version__ missing!'; "
        "assert triton.__version__ == '3.6.0', f'Wrong version: {triton.__version__}'; "
        "assert hasattr(triton, '__file__') and triton.__file__ is not None, 'triton.__file__ is None — namespace package!'; "
        "print(f'PYTHONPATH OK: __file__={triton.__file__}, __version__={triton.__version__}'); "
        "import triton.graph; "
        "print('triton.graph OK'); "
        "import triton.backends; "
        "be = list(triton.backends.backends.keys()) if hasattr(triton.backends, 'backends') else []; "
        "print(f'backends: {be}'); "
        "\"",
    )
    .env(KGIR_ENV)
)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App("triton-kgir-gpu-validation", image=triton_image)


# ---------------------------------------------------------------------------
# Helper: GPU pre-flight check
# ---------------------------------------------------------------------------
def _gpu_preflight(gpu_type: str) -> None:
    """Run GPU health checks; raise on failure with actionable messages."""
    import subprocess

    # 1. nvidia-smi
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"NVIDIA driver not available in container (nvidia-smi exit code {result.returncode}). "
            f"stderr: {result.stderr.strip()}"
        )
    print("=== nvidia-smi output ===")
    print(result.stdout)

    # 2. torch.cuda check
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA — torch.cuda.is_available() returned False.")

    device_count = torch.cuda.device_count()
    if device_count < 2:
        raise RuntimeError(f"Expected 2 GPUs, found {device_count}.")

    # 3. Log device details
    print(f"\n=== GPU Pre-flight ({gpu_type}) ===")
    print(f"  CUDA version (runtime) : {torch.version.cuda}")
    print(f"  PyTorch version        : {torch.__version__}")
    print(f"  Device count           : {device_count}")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        # PyTorch >= 2.0 uses 'total_memory'; older versions used 'total_mem'.
        total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        print(
            f"  GPU {i}: {props.name}  |  "
            f"sm_{props.major}{props.minor}  |  "
            f"{total_bytes / (1024**3):.1f} GB  |  "
            f"SMs={props.multi_processor_count}"
        )

    # Driver version via nvidia-smi (already confirmed working)
    drv_line = [ln for ln in result.stdout.splitlines() if "Driver Version" in ln]
    if drv_line:
        print(f"  {drv_line[0].strip()}")
    print("=== Pre-flight PASSED ===\n")


# ---------------------------------------------------------------------------
# Helper: run one pytest phase and return (phase_name, xml_bytes, summary_dict)
# ---------------------------------------------------------------------------
def _run_phase(
    gpu_type: str,
    phase_key: str,
) -> tuple[str, bytes, dict]:
    """Execute a single pytest phase; return (label, junit_xml_bytes, summary)."""
    import subprocess
    import xml.etree.ElementTree as ET

    phase_name, marker_expr = PHASES[phase_key]
    label = f"{gpu_type}_{phase_name}"
    xml_path = f"{RESULTS_DIR}/{label}.xml"

    # Collect only from graph test directories — the kernel_graph, multi_device,
    # and heterogeneous_hw markers are exclusively defined on graph tests.
    # Collecting from the full python/test/ tree triggers import errors in
    # existing Triton tests whose backend-discovery code path is incompatible
    # with the editable source build inside the container.
    test_dirs = [
        f"{TRITON_SRC}/python/test/unit/graph/",
        f"{TRITON_SRC}/python/test/integration/graph/",
    ]

    # We invoke pytest via ``python -c "import triton; ..."`` instead of
    # ``python -m pytest`` so that ``triton`` (and hence ``triton.backends``)
    # is *fully* initialised before pytest starts collecting test modules.
    # Without this, the collection-time import chain
    #   triton.graph.__init__ → dispatch → triton.backends._discover_backends()
    #     → amd/driver → triton.runtime → … → triton.backends  (circular!)
    # fails with ``ImportError: cannot import name 'backends' from partially
    # initialized module 'triton.backends'``.
    #
    # PYTHONPATH is set explicitly in the subprocess environment to guarantee
    # that ``/root/triton/python`` is the FIRST entry on ``sys.path``, ahead
    # of conda's ``site-packages``.  This prevents Python from resolving
    # ``triton`` as a namespace package if stale artifacts survived cleanup.
    #
    # ``--import-mode=importlib`` prevents a secondary issue where pytest
    # tries to import integration tests as ``graph.test_<name>`` (the bare
    # directory name) instead of resolving them via the filesystem.
    pytest_args = [
        *test_dirs,
        "-m", marker_expr,
        "-v", "--tb=short",
        f"--rootdir={TRITON_SRC}",
        f"--junitxml={xml_path}",
        "--import-mode=importlib",
    ]

    # Build a one-liner that pre-imports triton with diagnostics, then
    # delegates to pytest.main() with the arguments supplied via sys.argv.
    # The diagnostics help identify namespace-package interference if it
    # persists after cleanup.
    bootstrap = (
        "import sys; "
        "print(f'[bootstrap] sys.path[0:4] = {sys.path[0:4]}'); "
        "import triton; "
        "f = getattr(triton, '__file__', None); "
        "v = getattr(triton, '__version__', None); "
        "p = list(getattr(triton, '__path__', [])); "
        "print(f'[bootstrap] triton.__file__={f}'); "
        "print(f'[bootstrap] triton.__version__={v}'); "
        "print(f'[bootstrap] triton.__path__={p}'); "
        "assert f is not None, "
        "'triton.__file__ is None — namespace package detected! "
        "Stale triton artifacts in site-packages.'; "
        "assert v is not None, "
        "'triton.__version__ is None — broken triton package!'; "
        "import triton.backends; "
        "be = list(triton.backends.backends.keys()) "
        "if hasattr(triton.backends, 'backends') else []; "
        "print(f'[bootstrap] backends={be}'); "
        "import triton.graph; "
        "print('[bootstrap] triton.graph OK — starting pytest'); "
        "sys.exit(__import__('pytest').main(sys.argv[1:]))"
    )

    cmd = [sys.executable, "-c", bootstrap, *pytest_args]

    # Explicitly set PYTHONPATH in subprocess env as a safety belt.
    # The .env(KGIR_ENV) already sets it for the container, but
    # subprocess.run inherits from os.environ, and we want to be sure.
    sub_env = os.environ.copy()
    sub_env["PYTHONPATH"] = f"{TRITON_SRC}/python"

    print(f"\n>>> Running {label}: pytest -m \"{marker_expr}\"")
    proc = subprocess.run(cmd, capture_output=False, env=sub_env)
    print(f">>> pytest exit code: {proc.returncode}")

    # Read XML
    try:
        with open(xml_path, "rb") as f:
            xml_bytes = f.read()
    except FileNotFoundError:
        xml_bytes = b"<testsuites/>"

    # Parse summary
    summary: dict = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    try:
        root = ET.fromstring(xml_bytes)
        for ts in root.iter("testsuite"):
            summary["total"] += int(ts.get("tests", 0))
            summary["failed"] += int(ts.get("failures", 0))
            summary["errors"] += int(ts.get("errors", 0))
            summary["skipped"] += int(ts.get("skipped", 0))
        summary["passed"] = (
            summary["total"] - summary["failed"] - summary["errors"] - summary["skipped"]
        )
    except ET.ParseError:
        pass

    return label, xml_bytes, summary


# ---------------------------------------------------------------------------
# Helper: emit base64-delimited JUnit XML
# ---------------------------------------------------------------------------
def _emit_junit_xml(label: str, xml_bytes: bytes) -> None:
    """Print JUnit XML as base64 between delimiters."""
    encoded = base64.b64encode(xml_bytes).decode("ascii")
    print(f"\n===== JUNIT XML START {label} =====")
    print(encoded)
    print(f"===== JUNIT XML END {label} =====\n")


# ---------------------------------------------------------------------------
# Helper: print summary table
# ---------------------------------------------------------------------------
def _print_summary(gpu_type: str, results: list[tuple[str, dict]]) -> None:
    """Print a tabular summary of all phases."""
    header = f"{'Phase':<30} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Skip':>6} {'Error':>6}"
    sep = "-" * len(header)
    print(f"\n=== Summary ({gpu_type}) ===")
    print(header)
    print(sep)
    for label, s in results:
        print(
            f"{label:<30} {s['total']:>6} {s['passed']:>6} "
            f"{s['failed']:>6} {s['skipped']:>6} {s['errors']:>6}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Helper: Phase 3 heterogeneous skip note
# ---------------------------------------------------------------------------
def _phase3_note(gpu_type: str, summary: dict) -> None:
    """Print an explanatory note when Phase 3 tests skip on same-gen GPUs."""
    total = summary["total"]
    skipped = summary["skipped"]
    if skipped > 0:
        print(
            f"\nPhase 3: {skipped}/{total} heterogeneous tests skipped "
            f"(2x {gpu_type} same-generation). "
            "Cross-generation validation requires mixed GPU types "
            "(e.g., A100 + H100 in one container)."
        )


# ---------------------------------------------------------------------------
# Remote GPU functions — one per GPU type (Modal requires static gpu param)
# ---------------------------------------------------------------------------
@app.function(gpu="A100:2", timeout=REMOTE_TIMEOUT)
def run_tests_a100(phase: str = "all") -> list[tuple[str, bytes, dict]]:
    """Run KGIR test suite on 2x A100 GPUs."""
    return _run_all("A100", phase)


@app.function(gpu="H100:2", timeout=REMOTE_TIMEOUT)
def run_tests_h100(phase: str = "all") -> list[tuple[str, bytes, dict]]:
    """Run KGIR test suite on 2x H100 GPUs."""
    return _run_all("H100", phase)


def _run_all(gpu_type: str, phase: str) -> list[tuple[str, bytes, dict]]:
    """Core logic shared by both GPU functions."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Pre-flight
    _gpu_preflight(gpu_type)

    # Determine which phases to run
    if phase == "all":
        phase_keys = ["phase1", "phase2", "phase3"]
    else:
        if phase not in PHASES:
            raise ValueError(f"Unknown phase '{phase}'. Choose from: phase1, phase2, phase3, all")
        phase_keys = [phase]

    all_results: list[tuple[str, bytes, dict]] = []
    summary_rows: list[tuple[str, dict]] = []

    for pk in phase_keys:
        label, xml_bytes, summary = _run_phase(gpu_type, pk)
        all_results.append((label, xml_bytes, summary))
        summary_rows.append((label, summary))

        # Emit JUnit XML
        _emit_junit_xml(label, xml_bytes)

        # Phase 3 note
        if pk == "phase3":
            _phase3_note(gpu_type, summary)

    # Summary table
    _print_summary(gpu_type, summary_rows)

    return all_results


# ---------------------------------------------------------------------------
# Cross-architecture comparison
# ---------------------------------------------------------------------------
def _cross_architecture_comparison(
    a100_results: list[tuple[str, bytes, dict]],
    h100_results: list[tuple[str, bytes, dict]],
) -> None:
    """Parse JUnit XMLs from both GPU types and produce a comparison summary."""
    import xml.etree.ElementTree as ET

    def _extract_test_outcomes(results: list[tuple[str, bytes, dict]]) -> dict[str, str]:
        """Return {test_name: 'pass'|'fail'|'skip'|'error'} from JUnit XML bytes."""
        outcomes: dict[str, str] = {}
        for _label, xml_bytes, _summary in results:
            try:
                root = ET.fromstring(xml_bytes)
            except ET.ParseError:
                continue
            for tc in root.iter("testcase"):
                name = f"{tc.get('classname', '')}.{tc.get('name', '')}"
                if tc.find("failure") is not None:
                    outcomes[name] = "fail"
                elif tc.find("error") is not None:
                    outcomes[name] = "error"
                elif tc.find("skipped") is not None:
                    outcomes[name] = "skip"
                else:
                    outcomes[name] = "pass"
        return outcomes

    a100_map = _extract_test_outcomes(a100_results)
    h100_map = _extract_test_outcomes(h100_results)

    all_tests = sorted(set(a100_map.keys()) | set(h100_map.keys()))

    pass_a100_fail_h100: list[str] = []
    fail_a100_pass_h100: list[str] = []
    fail_both: list[str] = []
    pass_both: list[str] = []
    skip_both: list[str] = []
    other: list[str] = []

    for t in all_tests:
        a = a100_map.get(t, "missing")
        h = h100_map.get(t, "missing")
        if a == "pass" and h in ("fail", "error"):
            pass_a100_fail_h100.append(t)
        elif a in ("fail", "error") and h == "pass":
            fail_a100_pass_h100.append(t)
        elif a in ("fail", "error") and h in ("fail", "error"):
            fail_both.append(t)
        elif a == "pass" and h == "pass":
            pass_both.append(t)
        elif a == "skip" and h == "skip":
            skip_both.append(t)
        else:
            other.append(t)

    def _list_or_none(items: list[str]) -> str:
        return "\n  ".join(items) if items else "(none)"

    print("\n===== CROSS-ARCHITECTURE COMPARISON =====")
    print(f"Tests that PASS on A100 but FAIL on H100: [{len(pass_a100_fail_h100)}]")
    if pass_a100_fail_h100:
        print(f"  {_list_or_none(pass_a100_fail_h100)}")
    print(f"Tests that FAIL on A100 but PASS on H100: [{len(fail_a100_pass_h100)}]")
    if fail_a100_pass_h100:
        print(f"  {_list_or_none(fail_a100_pass_h100)}")
    print(f"Tests that FAIL on both: [{len(fail_both)}]")
    if fail_both:
        print(f"  {_list_or_none(fail_both)}")
    print(f"Tests that PASS on both: [{len(pass_both)}]")
    if pass_both:
        print(f"  {_list_or_none(pass_both)}")
    print(f"Tests that SKIP on both: [{len(skip_both)}]")
    if skip_both:
        print(f"  {_list_or_none(skip_both)}")
    if other:
        print(f"Other (asymmetric skip/missing): [{len(other)}]")
        print(f"  {_list_or_none(other)}")
    total_accounted = (
        len(pass_a100_fail_h100) + len(fail_a100_pass_h100)
        + len(fail_both) + len(pass_both) + len(skip_both) + len(other)
    )
    print(f"\nTotal unique tests: {len(all_tests)}  |  Accounted: {total_accounted}")
    print("==========================================\n")


# ---------------------------------------------------------------------------
# Local entrypoint — runs on the developer's machine
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    gpu_type: str = "A100",
    phase: str = "all",
) -> None:
    """
    Triton KGIR GPU validation entrypoint.

    Args:
        gpu_type: GPU architecture to test on. Choices: A100, H100.
        phase: Test phase to run. Choices: phase1, phase2, phase3, all.
    """
    gpu_type = gpu_type.upper()
    if gpu_type not in ("A100", "H100"):
        print(f"ERROR: --gpu-type must be A100 or H100, got '{gpu_type}'")
        sys.exit(1)
    if phase not in ("phase1", "phase2", "phase3", "all"):
        print(f"ERROR: --phase must be phase1, phase2, phase3, or all, got '{phase}'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Triton KGIR GPU Validation — {gpu_type} — {phase}")
    print(f"{'='*60}\n")

    # Dispatch to the correct remote function
    if gpu_type == "A100":
        results = run_tests_a100.remote(phase=phase)
    else:
        results = run_tests_h100.remote(phase=phase)

    # Re-print summary locally (results already printed in remote stdout)
    summary_rows = [(label, summary) for label, _xml, summary in results]
    _print_summary(gpu_type, summary_rows)

    print(f"\n{'='*60}")
    print(f"  {gpu_type} validation complete.")
    print(f"{'='*60}\n")
