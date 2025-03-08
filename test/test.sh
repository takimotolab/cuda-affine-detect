#!/bin/bash -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <cuda-gpu-arch> <cuda-include-path>"
    exit 1
fi

./cgeist.sh test1 $1 $2
diff <(../build/caf test1.mlir) <(cat test1.txt)
