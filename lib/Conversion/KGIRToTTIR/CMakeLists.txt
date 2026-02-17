add_triton_library(KGIRToTTIR
    KGIRToTTIRPass.cpp

    DEPENDS
    KGIRToTTIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonKGIRIR
    TritonIR
)
