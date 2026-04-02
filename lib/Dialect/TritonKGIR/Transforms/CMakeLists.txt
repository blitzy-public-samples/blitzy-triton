add_triton_library(TritonKGIRTransforms
  FusionAnalysis.cpp
  MemoryPlanning.cpp
  SchedulerPass.cpp

  DEPENDS
  TritonKGIRTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  TritonKGIRIR
  TritonIR
  TritonGPUIR
  MLIRIR
)
