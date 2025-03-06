#!/bin/bash -e

../Polygeist/build/bin/cgeist gemm.c -S -raise-scf-to-affine -o gemm.mlir
diff <(../build/caf gemm.mlir) <(cat gemm.txt)
