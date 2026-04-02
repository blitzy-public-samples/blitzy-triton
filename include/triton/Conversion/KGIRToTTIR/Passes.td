#ifndef TRITON_CONVERSION_KGIRTOTTIR_PASSES
#define TRITON_CONVERSION_KGIRTOTTIR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertKGIRToTTIR : Pass<"convert-kgir-to-ttir", "mlir::ModuleOp"> {
  let summary = "Convert KGIR to Triton TTIR";

  let description = [{
    This pass converts fused Triton Kernel Graph IR (KGIR) operations into
    valid Triton TTIR (Triton Intermediate Representation). It performs:

    - Producer-consumer kernel body merging with shared memory intermediate
      insertion for data exchange between fused kernels
    - Sibling kernel body merging with SM/CU partitioning for independent
      kernels sharing a launch
    - Unified grid computation for fused kernel launches
    - Per-target TTIR emission with target-specific tiling, shared memory
      allocation, and grid dimensions

    The emitted TTIR is standard Triton IR compilable by the existing
    unmodified single-kernel compilation pipeline for any supported backend.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::kgir::TritonKGIRDialect"];

  let options = [
    Option<"target", "target",
           "std::string", /*default*/"\"\"",
           "the GPU target for which to emit TTIR, e.g., cuda:80, hip:gfx942">,
  ];
}

#endif
