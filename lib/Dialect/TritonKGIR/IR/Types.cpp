//===- Types.cpp - TritonKGIR Type Implementations ----------*- C++ -*-===//
//
// Stub implementation to unblock module build. Will be replaced by the
// assigned agent with full type parsing/printing for HardwareProfileType,
// NodeMetadataType, and PerformanceAnnotationType.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonKGIR/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// MLIR's generated TableGen code for types with raw C++ 'double' parameters
// requires is_hashable_data<double> to be true so that hash_combine() can
// process double arguments directly in the generated hashKey() methods.
// HardwareProfileType has 3 double params and PerformanceAnnotationType has
// 6 double params.
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

// Generated type class implementations from TritonKGIRTypes.td.
#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonKGIR/IR/Types.cpp.inc"

// Register all KGIR custom types with the dialect.
void TritonKGIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonKGIR/IR/Types.cpp.inc"
      >();
}

// Custom assembly format for HardwareProfileType.
// Format: !ttkgir.hw_profile<vendor, arch, sm_count, smem, regs, gmem,
//                             bw, compute, warp, streams, interconnect, ibw>
mlir::Type HardwareProfileType::parse(mlir::AsmParser &odsParser) {
  if (odsParser.parseLess())
    return {};
  // parseString requires std::string*, not StringRef*
  std::string vendorStr, archGenerationStr, interconnectTypeStr;
  int32_t smCount, smemPerSmBytes, registersPerSm, warpSize,
      maxConcurrentStreams;
  int64_t globalMemoryBytes;
  double memoryBandwidthGbps, computeThroughputTflops,
      interconnectBandwidthGbps;
  if (odsParser.parseString(&vendorStr) || odsParser.parseComma() ||
      odsParser.parseString(&archGenerationStr) || odsParser.parseComma() ||
      odsParser.parseInteger(smCount) || odsParser.parseComma() ||
      odsParser.parseInteger(smemPerSmBytes) || odsParser.parseComma() ||
      odsParser.parseInteger(registersPerSm) || odsParser.parseComma() ||
      odsParser.parseInteger(globalMemoryBytes) || odsParser.parseComma() ||
      odsParser.parseFloat(memoryBandwidthGbps) || odsParser.parseComma() ||
      odsParser.parseFloat(computeThroughputTflops) ||
      odsParser.parseComma() ||
      odsParser.parseInteger(warpSize) || odsParser.parseComma() ||
      odsParser.parseInteger(maxConcurrentStreams) || odsParser.parseComma() ||
      odsParser.parseString(&interconnectTypeStr) || odsParser.parseComma() ||
      odsParser.parseFloat(interconnectBandwidthGbps) ||
      odsParser.parseGreater())
    return {};
  // std::string implicitly converts to StringRef for the get() parameters
  return get(odsParser.getContext(), vendorStr, archGenerationStr, smCount,
             smemPerSmBytes, registersPerSm, globalMemoryBytes,
             memoryBandwidthGbps, computeThroughputTflops, warpSize,
             maxConcurrentStreams, interconnectTypeStr,
             interconnectBandwidthGbps);
}

void HardwareProfileType::print(mlir::AsmPrinter &odsPrinter) const {
  // printFloat requires APFloat, not raw double
  odsPrinter << "<\"" << getVendor() << "\", \"" << getArchGeneration()
             << "\", " << getSmCount() << ", " << getSmemPerSmBytes() << ", "
             << getRegistersPerSm() << ", " << getGlobalMemoryBytes() << ", ";
  odsPrinter.printFloat(llvm::APFloat(getMemoryBandwidthGbps()));
  odsPrinter << ", ";
  odsPrinter.printFloat(llvm::APFloat(getComputeThroughputTflops()));
  odsPrinter << ", " << getWarpSize() << ", " << getMaxConcurrentStreams()
             << ", \"" << getInterconnectType() << "\", ";
  odsPrinter.printFloat(llvm::APFloat(getInterconnectBandwidthGbps()));
  odsPrinter << ">";
}

// Custom assembly format for NodeMetadataType.
// Format: !ttkgir.node_metadata<args, gx, gy, gz, smem, regs, reads, writes>
mlir::Type NodeMetadataType::parse(mlir::AsmParser &odsParser) {
  if (odsParser.parseLess())
    return {};
  int32_t numTensorArgs, gridDimX, gridDimY, gridDimZ, sharedMemoryBytes,
      registerPressure, numReads, numWrites;
  if (odsParser.parseInteger(numTensorArgs) || odsParser.parseComma() ||
      odsParser.parseInteger(gridDimX) || odsParser.parseComma() ||
      odsParser.parseInteger(gridDimY) || odsParser.parseComma() ||
      odsParser.parseInteger(gridDimZ) || odsParser.parseComma() ||
      odsParser.parseInteger(sharedMemoryBytes) || odsParser.parseComma() ||
      odsParser.parseInteger(registerPressure) || odsParser.parseComma() ||
      odsParser.parseInteger(numReads) || odsParser.parseComma() ||
      odsParser.parseInteger(numWrites) || odsParser.parseGreater())
    return {};
  return get(odsParser.getContext(), numTensorArgs, gridDimX, gridDimY,
             gridDimZ, sharedMemoryBytes, registerPressure, numReads,
             numWrites);
}

void NodeMetadataType::print(mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getNumTensorArgs() << ", " << getGridDimX() << ", "
             << getGridDimY() << ", " << getGridDimZ() << ", "
             << getSharedMemoryBytes() << ", " << getRegisterPressure() << ", "
             << getNumReads() << ", " << getNumWrites() << ">";
}

// Custom assembly format for PerformanceAnnotationType.
// Format: !ttkgir.perf_annotation<wc, mem, occ, bw, launch, transfer>
mlir::Type PerformanceAnnotationType::parse(mlir::AsmParser &odsParser) {
  if (odsParser.parseLess())
    return {};
  double wallClockUs, memoryThroughputGbps, occupancy, bandwidthUtilization,
      launchOverheadUs, transferTimeUs;
  if (odsParser.parseFloat(wallClockUs) || odsParser.parseComma() ||
      odsParser.parseFloat(memoryThroughputGbps) || odsParser.parseComma() ||
      odsParser.parseFloat(occupancy) || odsParser.parseComma() ||
      odsParser.parseFloat(bandwidthUtilization) || odsParser.parseComma() ||
      odsParser.parseFloat(launchOverheadUs) || odsParser.parseComma() ||
      odsParser.parseFloat(transferTimeUs) || odsParser.parseGreater())
    return {};
  return get(odsParser.getContext(), wallClockUs, memoryThroughputGbps,
             occupancy, bandwidthUtilization, launchOverheadUs,
             transferTimeUs);
}

void PerformanceAnnotationType::print(mlir::AsmPrinter &odsPrinter) const {
  // printFloat requires APFloat, not raw double
  odsPrinter << "<";
  odsPrinter.printFloat(llvm::APFloat(getWallClockUs()));
  odsPrinter << ", ";
  odsPrinter.printFloat(llvm::APFloat(getMemoryThroughputGbps()));
  odsPrinter << ", ";
  odsPrinter.printFloat(llvm::APFloat(getOccupancy()));
  odsPrinter << ", ";
  odsPrinter.printFloat(llvm::APFloat(getBandwidthUtilization()));
  odsPrinter << ", ";
  odsPrinter.printFloat(llvm::APFloat(getLaunchOverheadUs()));
  odsPrinter << ", ";
  odsPrinter.printFloat(llvm::APFloat(getTransferTimeUs()));
  odsPrinter << ">";
}
