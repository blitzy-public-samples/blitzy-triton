//===- Types.cpp - TritonKGIR Type Implementations ----------*- C++ -*-===//
//
// Part of the Triton project.
//
// Implements custom assembly format parsing and printing for the three
// TritonKGIR dialect types that use hasCustomAssemblyFormat = 1:
//
//   - HardwareProfileType  (12 parameters: vendor, arch, SM/CU specs,
//                           memory, compute, interconnect)
//   - NodeMetadataType     (8 parameters: tensor args, grid dims,
//                           shared memory, registers, reads, writes)
//   - PerformanceAnnotationType (6 parameters: wall-clock, throughput,
//                           occupancy, bandwidth, launch overhead, transfer)
//
// Also implements TritonKGIRDialect::registerTypes() which registers all
// KGIR types with the dialect, called from Dialect.cpp's initialize().
//
// The custom assembly format ensures MLIR textual IR round-trips correctly
// for all type parameters including strings (quoted), integers, and
// floating-point values (via AsmPrinter::printFloat for IEEE 754 fidelity).
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonKGIR/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h" // Required by Types.cpp.inc
#include "mlir/Support/LLVM.h"             // LLVM ADT utilities (StringRef)
#include "llvm/ADT/TypeSwitch.h"           // Required by Types.cpp.inc

// MLIR's generated TableGen code for types with raw C++ 'double' parameters
// uses llvm::hash_combine() inside the generated hashKey() methods for type
// storage uniquing. hash_combine() requires is_hashable_data<T> to be true
// for direct memcpy-based hashing. By default, is_hashable_data<double> is
// false because double has non-unique bit representations (+0.0/-0.0, NaN
// variants). Since our hardware profile and performance annotation types use
// double parameters for bandwidth, throughput, occupancy, etc., we must
// specialize this trait.
//
// This specialization is safe because:
// 1. We only hash doubles for type uniquing (storage identity), not for
//    numerical comparison
// 2. The HardwareProfile and PerformanceAnnotation types store exact
//    IEEE 754 values that should match bitwise for identity
// 3. Each translation unit that includes GET_TYPEDEF_CLASSES with double
//    parameters needs this specialization independently
namespace llvm {
namespace hashing {
namespace detail {
template <>
struct is_hashable_data<double> : std::true_type {};
} // namespace detail
} // namespace hashing
} // namespace llvm

using namespace mlir;
using namespace mlir::triton::kgir;

// Materialize generated type class implementations from TritonKGIRTypes.td.
// The generated Types.cpp.inc provides:
//   - TypeStorage subclasses with hashKey(), construct(), operator==()
//   - Factory methods: get(), getChecked()
//   - Accessor methods: getVendor(), getSmCount(), etc.
//   - Type dispatch infrastructure for the dialect's type parser/printer
//
// This MUST come before registerTypes() and the custom parse/print methods
// because the generated code defines the storage classes that the custom
// methods reference.
#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonKGIR/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// TritonKGIR Dialect — Type Registration
//===----------------------------------------------------------------------===//

/// Register all TritonKGIR custom types with the dialect. Called from
/// TritonKGIRDialect::initialize() in Dialect.cpp before operations and
/// attributes are registered, since operations reference these types in
/// their signatures.
///
/// The GET_TYPEDEF_LIST macro expands to the comma-separated list of all
/// type classes generated from TritonKGIRTypes.td:
///   HardwareProfileType, NodeMetadataType, PerformanceAnnotationType
void TritonKGIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonKGIR/IR/Types.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// HardwareProfileType — Custom Assembly Format
//===----------------------------------------------------------------------===//
//
// Encapsulates per-device hardware characteristics for the hardware-aware
// dispatch layer and adaptive cost model. All 12 fields from the concrete
// HardwareProfile schema (AAP §0.5.1) are serialized in a fixed positional
// format with quoted strings for vendor identifiers.
//
// Format:
//   !ttkgir.hw_profile<vendor, arch_generation, sm_count,
//                       smem_per_sm_bytes, registers_per_sm,
//                       global_memory_bytes, memory_bandwidth_gbps,
//                       compute_throughput_tflops, warp_size,
//                       max_concurrent_streams, interconnect_type,
//                       interconnect_bandwidth_gbps>
//
// Example (NVIDIA H100):
//   !ttkgir.hw_profile<"nvidia", "sm_90", 132, 232448, 65536,
//                       85899345920, 2.039000e+03, 9.894000e+02, 32, 128,
//                       "nvlink_4", 9.000000e+02>
//
// Example (AMD MI300X):
//   !ttkgir.hw_profile<"amd", "gfx942", 304, 65536, 65536,
//                       206158430208, 5.300000e+03, 1.307000e+03, 64, 128,
//                       "infinity_fabric", 8.960000e+02>
//

Type HardwareProfileType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Parse all 12 parameters in fixed positional order.
  // String parameters are quoted; integers and floats are unquoted.
  std::string vendor, archGeneration, interconnectType;
  int32_t smCount, smemPerSmBytes, registersPerSm, warpSize,
      maxConcurrentStreams;
  int64_t globalMemoryBytes;
  double memoryBandwidthGbps, computeThroughputTflops,
      interconnectBandwidthGbps;

  if (parser.parseString(&vendor) || parser.parseComma() ||
      parser.parseString(&archGeneration) || parser.parseComma() ||
      parser.parseInteger(smCount) || parser.parseComma() ||
      parser.parseInteger(smemPerSmBytes) || parser.parseComma() ||
      parser.parseInteger(registersPerSm) || parser.parseComma() ||
      parser.parseInteger(globalMemoryBytes) || parser.parseComma() ||
      parser.parseFloat(memoryBandwidthGbps) || parser.parseComma() ||
      parser.parseFloat(computeThroughputTflops) || parser.parseComma() ||
      parser.parseInteger(warpSize) || parser.parseComma() ||
      parser.parseInteger(maxConcurrentStreams) || parser.parseComma() ||
      parser.parseString(&interconnectType) || parser.parseComma() ||
      parser.parseFloat(interconnectBandwidthGbps) ||
      parser.parseGreater())
    return Type();

  // std::string implicitly converts to StringRef for the get() factory.
  // The MLIR type storage infrastructure copies string data into the
  // MLIRContext's allocator, so temporary std::string lifetime is safe.
  return HardwareProfileType::get(
      parser.getContext(), vendor, archGeneration, smCount, smemPerSmBytes,
      registersPerSm, globalMemoryBytes, memoryBandwidthGbps,
      computeThroughputTflops, warpSize, maxConcurrentStreams,
      interconnectType, interconnectBandwidthGbps);
}

void HardwareProfileType::print(AsmPrinter &printer) const {
  // Print string parameters quoted, integers directly, and doubles via
  // printFloat(APFloat) to ensure IEEE 754 round-trip fidelity. The
  // printFloat method uses MLIR's canonical floating-point format which
  // is always parseable by parseFloat(double &).
  printer << "<\"" << getVendor() << "\", \"" << getArchGeneration()
          << "\", " << getSmCount() << ", " << getSmemPerSmBytes() << ", "
          << getRegistersPerSm() << ", " << getGlobalMemoryBytes() << ", ";
  printer.printFloat(llvm::APFloat(getMemoryBandwidthGbps()));
  printer << ", ";
  printer.printFloat(llvm::APFloat(getComputeThroughputTflops()));
  printer << ", " << getWarpSize() << ", " << getMaxConcurrentStreams()
          << ", \"" << getInterconnectType() << "\", ";
  printer.printFloat(llvm::APFloat(getInterconnectBandwidthGbps()));
  printer << ">";
}

//===----------------------------------------------------------------------===//
// NodeMetadataType — Custom Assembly Format
//===----------------------------------------------------------------------===//
//
// Per-kernel-node metadata capturing grid dimensions, resource usage, and
// memory access characteristics for fusion analysis and scheduling decisions.
// All 8 fields are int32_t representing kernel launch configuration and
// resource consumption.
//
// Format:
//   !ttkgir.node_metadata<num_tensor_args, grid_dim_x, grid_dim_y,
//                          grid_dim_z, shared_memory_bytes,
//                          register_pressure, num_reads, num_writes>
//
// Example (matmul kernel with 4 tensor args, 1D grid of 1024 blocks):
//   !ttkgir.node_metadata<4, 1024, 1, 1, 49152, 64, 3, 1>
//
// Example (elementwise kernel with 2 tensor args, 2D grid):
//   !ttkgir.node_metadata<2, 256, 4, 1, 0, 24, 1, 1>
//

Type NodeMetadataType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Parse all 8 int32_t parameters in fixed positional order.
  int32_t numTensorArgs, gridDimX, gridDimY, gridDimZ, sharedMemoryBytes,
      registerPressure, numReads, numWrites;

  if (parser.parseInteger(numTensorArgs) || parser.parseComma() ||
      parser.parseInteger(gridDimX) || parser.parseComma() ||
      parser.parseInteger(gridDimY) || parser.parseComma() ||
      parser.parseInteger(gridDimZ) || parser.parseComma() ||
      parser.parseInteger(sharedMemoryBytes) || parser.parseComma() ||
      parser.parseInteger(registerPressure) || parser.parseComma() ||
      parser.parseInteger(numReads) || parser.parseComma() ||
      parser.parseInteger(numWrites) || parser.parseGreater())
    return Type();

  return NodeMetadataType::get(parser.getContext(), numTensorArgs, gridDimX,
                               gridDimY, gridDimZ, sharedMemoryBytes,
                               registerPressure, numReads, numWrites);
}

void NodeMetadataType::print(AsmPrinter &printer) const {
  // All parameters are integers — direct printing is safe and round-trips.
  printer << "<" << getNumTensorArgs() << ", " << getGridDimX() << ", "
          << getGridDimY() << ", " << getGridDimZ() << ", "
          << getSharedMemoryBytes() << ", " << getRegisterPressure() << ", "
          << getNumReads() << ", " << getNumWrites() << ">";
}

//===----------------------------------------------------------------------===//
// PerformanceAnnotationType — Custom Assembly Format
//===----------------------------------------------------------------------===//
//
// Stores measured performance data per kernel per target for the closed-loop
// feedback mechanism. The runtime profiler writes measured data back into
// KGIR node annotations after each execution iteration, enabling the
// feedback controller to compare predictions vs actuals and trigger
// re-optimization when prediction error exceeds the configured sensitivity
// threshold (TRITON_FEEDBACK_SENSITIVITY, default 0.15).
//
// All 6 fields are doubles to accommodate fractional measurements from
// GPU event timing APIs (CUDA events, HIP events).
//
// Format:
//   !ttkgir.perf_annotation<wall_clock_us, memory_throughput_gbps,
//                            occupancy, bandwidth_utilization,
//                            launch_overhead_us, transfer_time_us>
//
// Example (single-device kernel with good occupancy):
//   !ttkgir.perf_annotation<1.253000e+02, 1.200500e+03, 8.500000e-01,
//                            7.200000e-01, 5.200000e+00, 0.000000e+00>
//
// Example (cross-device kernel with transfer overhead):
//   !ttkgir.perf_annotation<3.450000e+02, 8.500000e+02, 0.000000e+00,
//                            0.000000e+00, 4.100000e+00, 2.150000e+02>
//

Type PerformanceAnnotationType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Parse all 6 double parameters in fixed positional order.
  double wallClockUs, memoryThroughputGbps, occupancy, bandwidthUtilization,
      launchOverheadUs, transferTimeUs;

  if (parser.parseFloat(wallClockUs) || parser.parseComma() ||
      parser.parseFloat(memoryThroughputGbps) || parser.parseComma() ||
      parser.parseFloat(occupancy) || parser.parseComma() ||
      parser.parseFloat(bandwidthUtilization) || parser.parseComma() ||
      parser.parseFloat(launchOverheadUs) || parser.parseComma() ||
      parser.parseFloat(transferTimeUs) || parser.parseGreater())
    return Type();

  return PerformanceAnnotationType::get(
      parser.getContext(), wallClockUs, memoryThroughputGbps, occupancy,
      bandwidthUtilization, launchOverheadUs, transferTimeUs);
}

void PerformanceAnnotationType::print(AsmPrinter &printer) const {
  // Use printFloat(APFloat) for all double parameters to ensure IEEE 754
  // round-trip fidelity. This produces canonical scientific notation (e.g.,
  // "1.253000e+02") that parseFloat(double &) handles correctly, avoiding
  // precision loss from default C++ double-to-string formatting.
  printer << "<";
  printer.printFloat(llvm::APFloat(getWallClockUs()));
  printer << ", ";
  printer.printFloat(llvm::APFloat(getMemoryThroughputGbps()));
  printer << ", ";
  printer.printFloat(llvm::APFloat(getOccupancy()));
  printer << ", ";
  printer.printFloat(llvm::APFloat(getBandwidthUtilization()));
  printer << ", ";
  printer.printFloat(llvm::APFloat(getLaunchOverheadUs()));
  printer << ", ";
  printer.printFloat(llvm::APFloat(getTransferTimeUs()));
  printer << ">";
}
