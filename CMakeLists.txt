cmake_minimum_required(VERSION 3.20.0)
project(cuda-affine-detect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include(AddLLVM)
include(TableGen)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS} ${MLIR_LIBRARY_DIRS})

set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_executable(caf src/main.cpp)
target_link_libraries(caf
  PRIVATE
    ${dialect_libs}
    ${extension_libs}
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
)
