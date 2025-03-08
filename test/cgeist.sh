#!/bin/bash -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <filename> <cuda-gpu-arch> <cuda-include-path>"
    exit 1
fi

../Polygeist/build/bin/cgeist $1.cu \
    --cuda-gpu-arch=$2 \
    -nocudalib \
    -nocudainc \
    -I ../Polygeist/llvm-project/build/lib/clang/18/include/ \
    -I $3 \
    --function=* \
    -S \
    -raise-scf-to-affine \
    -o $1.mlir
