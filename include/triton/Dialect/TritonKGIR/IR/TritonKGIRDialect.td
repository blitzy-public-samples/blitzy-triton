#ifndef TRITONKGIR_DIALECT
#define TRITONKGIR_DIALECT

include "mlir/IR/OpBase.td"

def TritonKGIR_Dialect : Dialect {
  let name = "ttkgir";

  let cppNamespace = "::mlir::triton::kgir";

  let description = [{
    Triton Kernel Graph IR (KGIR) Dialect.

    Models a directed acyclic graph (DAG) of kernel launches for cross-kernel
    optimization, including fusion analysis, memory planning, scheduling, and
    multi-target hardware-aware dispatch. KGIR nodes are target-agnostic at
    construction and mutable for runtime annotation write-back.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif // TRITONKGIR_DIALECT
