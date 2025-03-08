# CUDA Affine Detect

## What is this?

CUDAコードをLLVM MLIR上で解析し、affineを持つカーネルのリストおよびLLVM IRを出力するプログラム。

## Build

```sh
# 本リポジトリをクローン
git clone --recursive https://github.com/takimotolab/cuda-affine-detect.git
cd cuda-affine-detect

# LLVMをビルド
mkdir Polygeist/llvm-project/build
cd Polygeist/llvm-project/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DMLIR_ENABLE_CUDA_RUNNER=1 \
    -DCUDA_CXX=/path/to/cuda/bin/nvcc \
    -DCUDA_PATH=/path/to/cuda \
    -DCUDA_CMAKE_COMPILER=/path/to/cuda/bin/nvcc
ninja

# Polygeistをビルド
mkdir ../../build
cd ../../build
cmake -G Ninja ../ \
    -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
    -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DPOLYGEIST_ENABLE_CUDA=1
ninja

# cuda-affine-detectをビルド
mkdir ../../build
cd ../../build
cmake -G Ninja ../ \
    -DLLVM_DIR=$PWD/../Polygeist/llvm-project/build/lib/cmake/llvm \
    -DMLIR_DIR=$PWD/../Polygeist/llvm-project/build/lib/cmake/mlir
ninja
```

## Caution

LLVMをビルドしたらディレクトリを別の場所へ移動しないこと。
LLVMのビルドディレクトリ中の.cmakeファイルにフルパスが記録されている都合上、参照が壊れるため。
