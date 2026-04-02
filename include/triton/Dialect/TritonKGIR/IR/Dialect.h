//===- Dialect.h - TritonKGIR Dialect Aggregate Header -----------*- C++ -*-===//
//
// Part of the Triton project.
//
// Aggregate header for the TritonKGIR (Kernel Graph IR) dialect.
// This is the single include point for consumers of the KGIR dialect,
// providing the dialect class, custom types, attribute definitions, and
// operation definitions. All declarations are generated from TableGen
// definitions under TritonKGIR/IR/*.td via mlir_tablegen.
//
// Include ordering is critical:
//   1. Dependent dialect headers (Triton, TritonGPU)
//   2. Generated dialect class (Dialect.h.inc)
//   3. Generated type classes (Types.h.inc)
//   4. Generated attribute classes (TritonKGIRAttrDefs.h.inc)
//   5. Generated operation classes (Ops.h.inc)
//
//===----------------------------------------------------------------------===//

#pragma once

// TritonKGIR depends on the Triton core dialect and the TritonGPU dialect.
// TritonKGIRDialect.td declares:
//   dependentDialects = ["triton::TritonDialect",
//                        "triton::gpu::TritonGPUDialect"]
// These headers must precede KGIR generated fragments so that dependent
// type registries and dialect infrastructure are fully available.
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// Generated dialect class declaration for ttkgir (TritonKGIRDialect in
// the ::mlir::triton::kgir namespace). Produced by:
//   mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=ttkgir)
#include "triton/Dialect/TritonKGIR/IR/Dialect.h.inc"

// Generated type class declarations: HardwareProfileType,
// NodeMetadataType, PerformanceAnnotationType. Produced by:
//   mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=ttkgir)
#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonKGIR/IR/Types.h.inc"

// Generated attribute class declarations: MemoryAccessPatternAttr,
// HardwareTargetAnnotationAttr, RuntimePerformanceAttr, FusionDecisionAttr.
// Produced by:
//   mlir_tablegen(TritonKGIRAttrDefs.h.inc -gen-attrdef-decls)
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonKGIR/IR/TritonKGIRAttrDefs.h.inc"

// Generated operation class declarations: KernelLaunchOp, DataDepOp,
// AntiDepOp, TransferOp, FusedKernelOp, GraphOp. Produced by:
//   mlir_tablegen(Ops.h.inc -gen-op-decls)
#define GET_OP_CLASSES
#include "triton/Dialect/TritonKGIR/IR/Ops.h.inc"
