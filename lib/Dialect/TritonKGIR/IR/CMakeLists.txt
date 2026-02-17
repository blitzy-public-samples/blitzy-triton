add_triton_library(TritonKGIRIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
    TritonKGIRTableGen

  LINK_LIBS PUBLIC
    MLIRIR
    TritonIR
    TritonGPUIR
)
