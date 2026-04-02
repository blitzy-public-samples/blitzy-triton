//===- Dialect.cpp - TritonKGIR Dialect Registration -----------*- C++ -*-===//
//
// Part of the Triton project.
//
// Implements TritonKGIRDialect::initialize() which registers all custom types,
// operations, attributes, and interfaces for the TritonKGIR (Kernel Graph IR)
// dialect. This makes the dialect usable by the MLIR infrastructure for
// graph-level cross-kernel optimization including fusion analysis, memory
// planning, scheduling, and multi-target hardware-aware dispatch.
//
// Registration ordering is critical:
//   1. registerTypes() — Types must be registered before operations that
//      reference them (HardwareProfileType, NodeMetadataType,
//      PerformanceAnnotationType)
//   2. addOperations() — Operations reference types in their signatures
//      (KernelLaunchOp, DataDepOp, AntiDepOp, TransferOp, FusedKernelOp,
//      GraphOp)
//   3. addAttributes() — Attributes used by operations
//      (MemoryAccessPatternAttr, HardwareTargetAnnotationAttr,
//      RuntimePerformanceAttr, FusionDecisionAttr)
//   4. addInterfaces() — Dialect interface additions
//      (TritonInlinerInterface)
//
// Follows the combined pattern of:
//   - Triton/IR/Dialect.cpp (registerTypes → addOperations → addInterfaces)
//   - Gluon/IR/Dialect.cpp (addAttributes with GET_ATTRDEF_CLASSES/LIST)
//   - TritonInstrument/IR/Dialect.cpp (addInterfaces<TritonInlinerInterface>)
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonKGIR/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "llvm/ADT/TypeSwitch.h"

// MLIR's generated TableGen code for attributes with raw C++ 'double'
// parameters requires two infrastructure extensions not provided by default
// MLIR/LLVM headers:
//
// 1. llvm::hashing::detail::is_hashable_data<double> must be true so that
//    llvm::hash_combine() can process double arguments directly. Without this,
//    get_hashable_data<double> falls through to hash_value(double) which does
//    not exist in the default LLVM overload set, and the using-declaration in
//    get_hashable_data only captures overloads visible at template definition
//    time (in the header), not at instantiation time in this .cpp file.
//
// 2. mlir::FieldParser<double> must be specialized so that the generated
//    assemblyFormat parsing code can parse floating-point literals.
//
// These are needed by RuntimePerformanceAttr (4 double params) and
// FusionDecisionAttr (1 double param) from TritonKGIRAttrDefs.td.

namespace llvm {
namespace hashing {
namespace detail {
/// Specialize is_hashable_data for double so that hash_combine() can
/// directly memcpy double values into its hash buffer. This is safe because
/// double is a fixed-size (8 byte) trivially-copyable type whose raw IEEE 754
/// bit representation is suitable for hashing.
template <>
struct is_hashable_data<double> : std::true_type {};
} // namespace detail
} // namespace hashing
} // namespace llvm

namespace mlir {
/// FieldParser specialization for raw C++ double type.
/// Parses a floating-point literal from the MLIR assembly format.
template <>
struct FieldParser<double> {
  static FailureOr<double> parse(AsmParser &parser) {
    double result;
    if (parser.parseFloat(result))
      return failure();
    return result;
  }
};
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::kgir;

// Generated dialect class implementation infrastructure produced by
// mlir_tablegen from TritonKGIRDialect.td. Provides the auto-generated
// TritonKGIRDialect class implementation (constructor, destructor, dialect
// name accessor, etc.) in the ::mlir::triton::kgir namespace.
#include "triton/Dialect/TritonKGIR/IR/Dialect.cpp.inc"

// Generated attribute class implementations produced by mlir_tablegen from
// TritonKGIRAttrDefs.td. Materializes the full class bodies for:
//   - MemoryAccessPatternAttr (per-tensor read/write/readwrite patterns)
//   - HardwareTargetAnnotationAttr (device/vendor/arch/stream assignments)
//   - RuntimePerformanceAttr (measured wall-clock, throughput, occupancy)
//   - FusionDecisionAttr (fusion accept/reject with rationale per target)
// Including parse/print methods, storage class, and accessor implementations.
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonKGIR/IR/TritonKGIRAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TritonKGIRDialect::initialize
//===----------------------------------------------------------------------===//

void TritonKGIRDialect::initialize() {
  // Register custom types first. Types must be available before operations
  // that reference them in their signatures. The registerTypes() method is
  // declared via extraClassDeclaration in TritonKGIRDialect.td and
  // implemented in Types.cpp, registering:
  //   - HardwareProfileType (per-device SM count, SMEM, registers, bandwidth)
  //   - NodeMetadataType (memory patterns, shapes, grid, resource usage)
  //   - PerformanceAnnotationType (per-target measured metrics)
  registerTypes();

  // Register all 6 KGIR operations. These are structural/metadata operations
  // representing a DAG of kernel launches — they do NOT model tensor
  // computation. Operation class declarations come from Ops.h.inc (included
  // via Dialect.h); full class implementations are in Ops.cpp.
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonKGIR/IR/Ops.cpp.inc"
      >();

  // Register all 4 KGIR attributes used to annotate nodes and edges with
  // memory access patterns, hardware target metadata, runtime performance
  // measurements, and fusion decision information. Attribute class
  // implementations are materialized above via GET_ATTRDEF_CLASSES.
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonKGIR/IR/TritonKGIRAttrDefs.cpp.inc"
      >();

  // Register the Triton inliner interface for consistency with all existing
  // Triton dialects (Triton, Gluon, TritonInstrument all register this).
  // The base TritonInlinerInterface behavior from Interfaces.h is sufficient
  // for KGIR — no custom inliner overrides are needed since KGIR models
  // graph semantics rather than inlineable compute operations.
  addInterfaces<TritonInlinerInterface>();
}
