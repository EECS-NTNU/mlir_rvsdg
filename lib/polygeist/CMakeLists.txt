add_mlir_dialect_library(MLIRPolygeist
Dialect.cpp
Ops.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/polygeist

DEPENDS
MLIRPolygeistOpsIncGen

LINK_LIBS PUBLIC
MLIRIR
MLIRMemRefDialect
MLIRLLVMDialect
MLIROpenMPDialect
MLIRAffineDialect
MLIRSupport
MLIRSCFTransforms
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/polygeist" DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/polygeist" DESTINATION include
    FILES_MATCHING PATTERN "*.td"
)

install(DIRECTORY "${CMAKE_BINARY_DIR}/include/polygeist" DESTINATION include
    FILES_MATCHING PATTERN "*.h.inc"
)
