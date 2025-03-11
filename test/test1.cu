#include "cuda.h"

#define N 20

extern "C" {
  __global__ void bar(double *w) {
    w[threadIdx.x] = 2.0;
  }
}

extern "C" {
  __global__ void scaleVector(float *vector, float scalar, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
      return;
    }
    for (; i < size; ++i) {
      vector[i] *= scalar;
    }
  }
}

extern "C" {
  __global__ void badKernel(float *vector, float scalar, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = (i * 7) % size;
    if (i >= size || n >= size) {
      return;
    }
    for (; i < size; i += n) {
      vector[i] *= scalar;
    }
  }
}

extern "C" {
  void baz(double *w) {
    bar<<<1, N>>>(w);
  }
}

extern "C" {
  void hostScaleVector(float *vector, float scalar, int size) {
    scaleVector<<<1, N>>>(vector, scalar, size);
  }
}

extern "C" {
  void hostBadKernel(float *vector, float scalar, int size) {
    badKernel<<<1, N>>>(vector, scalar, size);
  }
}

/*
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @__device_stub__bar(%arg0: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %cst = arith.constant 2.000000e+00 : f64
    %0 = gpu.thread_id  x
    affine.store %cst, %arg0[symbol(%0)] : memref<?xf64>
    return
  }
  func.func private @__device_stub__scaleVector(%arg0: memref<?xf32>, %arg1: f32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = gpu.block_id  x
    %2 = arith.index_cast %1 : index to i32
    %3 = gpu.block_dim  x
    %4 = arith.index_cast %3 : index to i32
    %5 = arith.muli %2, %4 : i32
    %6 = gpu.thread_id  x
    %7 = arith.index_cast %6 : index to i32
    %8 = arith.addi %5, %7 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = arith.cmpi slt, %8, %arg2 : i32
    scf.if %10 {
      affine.for %arg3 = %9 to %0 {
        %11 = affine.load %arg0[%arg3] : memref<?xf32>
        %12 = arith.mulf %11, %arg1 : f32
        affine.store %12, %arg0[%arg3] : memref<?xf32>
      }
    }
    return
  }
  func.func private @__device_stub__badKernel(%arg0: memref<?xf32>, %arg1: f32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %true = arith.constant true
    %c7_i32 = arith.constant 7 : i32
    %0 = gpu.block_id  x
    %1 = arith.index_cast %0 : index to i32
    %2 = gpu.block_dim  x
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.muli %1, %3 : i32
    %5 = gpu.thread_id  x
    %6 = arith.index_cast %5 : index to i32
    %7 = arith.addi %4, %6 : i32
    %8 = arith.muli %7, %c7_i32 : i32
    %9 = arith.remsi %8, %arg2 : i32
    %10 = arith.cmpi sge, %7, %arg2 : i32
    %11 = scf.if %10 -> (i1) {
      scf.yield %true : i1
    } else {
      %13 = arith.cmpi sge, %9, %arg2 : i32
      scf.yield %13 : i1
    }
    %12 = arith.xori %11, %true : i1
    scf.if %12 {
      %13 = arith.index_cast %arg2 : i32 to index
      %14 = arith.index_cast %7 : i32 to index
      %15 = arith.index_cast %9 : i32 to index
      scf.for %arg3 = %14 to %13 step %15 {
        %16 = arith.subi %arg3, %14 : index
        %17 = arith.divui %16, %15 : index
        %18 = arith.muli %17, %15 : index
        %19 = arith.addi %14, %18 : index
        %20 = memref.load %arg0[%19] : memref<?xf32>
        %21 = arith.mulf %20, %arg1 : f32
        memref.store %21, %arg0[%19] : memref<?xf32>
      }
    }
    return
  }
  func.func @baz(%arg0: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %c1, %arg8 = %c1, %arg9 = %c1) threads(%arg4, %arg5, %arg6) in (%arg10 = %c20, %arg11 = %c1, %arg12 = %c1) {
      func.call @__device_stub__bar(%arg0) : (memref<?xf64>) -> ()
      gpu.terminator
    }
    return
  }
  func.func @hostScaleVector(%arg0: memref<?xf32>, %arg1: f32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c20, %arg13 = %c1, %arg14 = %c1) {
      func.call @__device_stub__scaleVector(%arg0, %arg1, %arg2) : (memref<?xf32>, f32, i32) -> ()
      gpu.terminator
    }
    return
  }
  func.func @hostBadKernel(%arg0: memref<?xf32>, %arg1: f32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c20, %arg13 = %c1, %arg14 = %c1) {
      func.call @__device_stub__badKernel(%arg0, %arg1, %arg2) : (memref<?xf32>, f32, i32) -> ()
      gpu.terminator
    }
    return
  }
}
*/
