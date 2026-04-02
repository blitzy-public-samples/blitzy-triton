set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonKGIROps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TritonKGIROps TritonKGIROps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TritonKGIRDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=ttkgir)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=ttkgir)
add_mlir_doc(TritonKGIRDialect TritonKGIRDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TritonKGIRTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=ttkgir)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=ttkgir)

set(LLVM_TARGET_DEFINITIONS TritonKGIRAttrDefs.td)
mlir_tablegen(TritonKGIRAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonKGIRAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(TritonKGIRTableGen)
