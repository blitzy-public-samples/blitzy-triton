# -*- Python -*-
# ruff: noqa: F821

import os

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner for KGIR kernel graph dialect tests.
#
# This config establishes the TRITON_KGIR test suite for KGIR MLIR dialect
# tests. It can be invoked in two ways:
#   1. Discovered as a nested suite by the root 'lit <build_dir>/test/'
#   2. Invoked directly via 'lit <build_dir>/test/KernelGraph/'
#
# In both cases, build variables (triton_obj_root, llvm_tools_dir, etc.) must
# be loaded from the parent test suite's generated lit.site.cfg.py.

# Ensure build configuration variables are available. When this config is
# loaded without a corresponding lit.site.cfg.py in the build tree (no
# KernelGraph-specific site config exists), we locate and load the parent
# test suite's generated site configuration to obtain the build variables.
if not hasattr(config, 'triton_obj_root'):
    import glob

    import lit.llvm
    lit.llvm.initialize(lit_config, config)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_test_dir = os.path.dirname(this_dir)

    # First, check if a generated lit.site.cfg.py exists alongside the parent
    # test directory (when running from a build tree that mirrors source layout).
    parent_site_cfg = os.path.join(parent_test_dir, 'lit.site.cfg.py')
    if not os.path.exists(parent_site_cfg):
        # Search common build tree locations relative to the repository root.
        source_root = os.path.dirname(parent_test_dir)
        candidates = sorted(glob.glob(os.path.join(
            source_root, 'build', '*', 'test', 'lit.site.cfg.py'
        )))
        if candidates:
            parent_site_cfg = candidates[-1]
        else:
            lit_config.fatal(
                "Could not find lit.site.cfg.py in the build tree. "
                "Build the project first, then run: "
                "'lit <build_dir>/test/KernelGraph/'"
            )

    # Load the parent site config. This sets build variables on the config
    # object and subsequently loads the root test/lit.cfg.py. Our KGIR-specific
    # overrides below take precedence over the root configuration.
    lit_config.load_config(config, parent_site_cfg)

# (config is an instance of TestingConfig created when discovering tests)
# name: The name of this test suite — scoped to the KGIR kernel graph dialect.
config.name = 'TRITON_KGIR'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
# KGIR tests are MLIR-only — no LLVM IR (.ll) tests in this subdirectory.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
# Resolves to the KernelGraph subdirectory containing this config file.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
# Scoped to the KernelGraph subdirectory within the build tree.
config.test_exec_root = os.path.join(config.triton_obj_root, 'test', 'KernelGraph')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(("%shlibdir", config.llvm_shlib_dir))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

config.triton_tools_dir = os.path.join(config.triton_obj_root, 'bin')
config.filecheck_dir = os.path.join(config.triton_obj_root, 'bin', 'FileCheck')

# FileCheck -enable-var-scope is enabled by default in MLIR test.
# This option avoids accidentally reusing variables across -LABEL match;
# it can be explicitly opted-in by prefixing the variable name with $.
config.environment["FILECHECK_OPTS"] = "--enable-var-scope"

tool_dirs = [config.triton_tools_dir, config.llvm_tools_dir, config.filecheck_dir]

# Tweak the PATH to include the tools dir.
for d in tool_dirs:
    llvm_config.with_environment('PATH', d, append_path=True)

tools = [
    'triton-opt',
    'triton-llvm-opt',
    'mlir-translate',
    'llc',
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
]

# Static libraries are not built if LLVM_BUILD_SHARED_LIBS is ON.
if config.build_shared_libs:
    config.available_features.add("shared-libs")

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.mlir_binary_dir, 'python_packages', 'triton'),
], append_path=True)
