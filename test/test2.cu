#include "cuda.h"

#include <stdio.h>

#define N 20

extern "C" {
  __global__ void f(float *v1, float *v2, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; ++i) {
      v1[i] = v1[i] + v2[i];
    }
  }
}

extern "C" {
  __global__ void g(float *v1, float *v2, float *v3, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; ++i) {
      v2[i] = v1[i] * v3[i];
    }
  }
}

extern "C" {
  __global__ void h(float *v1, float *v2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = (i * 7) % n;
    if (i >= n || s >= n) {
      return;
    }
    for (; i < n; i += s) {
      v1[i] = v1[i] + v2[i];
    }
  }
}

extern "C" {
  void host1(float *v1, float *v2, float *v3, int n, bool b) {
    f<<<1, N>>>(v1, v2, n);
    if (b) {
      printf("true");
    } else {
      printf("false");
    }
    g<<<1, N>>>(v1, v2, v3, n);
  }
}

extern "C" {
  void host2(float *v1, float *v2, float *v3, int n, bool b) {
    if (b) {
      f<<<1, N>>>(v1, v2, n);
    } else {
      printf("false");
    }
    g<<<1, N>>>(v1, v2, v3, n);
  }
}

extern "C" {
  void host3(float *v1, float *v2, float *v3, int n, bool b) {
    f<<<1, N>>>(v1, v2, n);
    if (b) {
      g<<<1, N>>>(v1, v2, v3, n);
    } else {
      printf("false");
    }
  }
}

extern "C" {
  void host4(float *v1, float *v2, float *v3, int n, bool b1, bool b2) {
    if (b1) {
      f<<<1, N>>>(v1, v2, n);
    } else {
      printf("false");
    }
    g<<<1, N>>>(v1, v2, v3, n);
    if (b2) {
      h<<<1, N>>>(v1, v2, n);
    } else {
      printf("false");
    }
  }
}

extern "C" {
  void host5(float *v1, float *v2, float *v3, int n, bool b1, bool b2) {
    if (b1) {
      h<<<1, N>>>(v1, v2, n);
    } else {
      printf("false");
    }
    g<<<1, N>>>(v1, v2, v3, n);
    if (b2) {
      f<<<1, N>>>(v1, v2, n);
    } else {
      printf("false");
    }
  }
}

/*
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str1("false\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("true\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func private @__device_stub__f(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %0 = gpu.block_id  x
    %1 = arith.index_cast %0 : index to i32
    %2 = gpu.block_dim  x
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.muli %1, %3 : i32
    %5 = gpu.thread_id  x
    %6 = arith.index_cast %5 : index to i32
    %7 = arith.addi %4, %6 : i32
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.index_cast %7 : i32 to index
    affine.for %arg3 = %9 to %8 {
      %10 = affine.load %arg0[%arg3] : memref<?xf32>
      %11 = affine.load %arg1[%arg3] : memref<?xf32>
      %12 = arith.addf %10, %11 : f32
      affine.store %12, %arg0[%arg3] : memref<?xf32>
    }
    return
  }
  func.func private @__device_stub__g(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %0 = gpu.block_id  x
    %1 = arith.index_cast %0 : index to i32
    %2 = gpu.block_dim  x
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.muli %1, %3 : i32
    %5 = gpu.thread_id  x
    %6 = arith.index_cast %5 : index to i32
    %7 = arith.addi %4, %6 : i32
    %8 = arith.index_cast %arg3 : i32 to index
    %9 = arith.index_cast %7 : i32 to index
    affine.for %arg4 = %9 to %8 {
      %10 = affine.load %arg0[%arg4] : memref<?xf32>
      %11 = affine.load %arg2[%arg4] : memref<?xf32>
      %12 = arith.mulf %10, %11 : f32
      affine.store %12, %arg1[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @__device_stub__h(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
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
        %21 = memref.load %arg1[%19] : memref<?xf32>
        %22 = arith.addf %20, %21 : f32
        memref.store %22, %arg0[%19] : memref<?xf32>
      }
    }
    return
  }
  func.func @host1(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i8) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0_i8 = arith.constant 0 : i8
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c20, %arg15 = %c1, %arg16 = %c1) {
      func.call @__device_stub__f(%arg0, %arg1, %arg3) : (memref<?xf32>, memref<?xf32>, i32) -> ()
      gpu.terminator
    }
    %0 = arith.cmpi ne, %arg4, %c0_i8 : i8
    scf.if %0 {
      %1 = llvm.mlir.addressof @str0 : !llvm.ptr
      %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
      %3 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    } else {
      %1 = llvm.mlir.addressof @str1 : !llvm.ptr
      %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %3 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c20, %arg15 = %c1, %arg16 = %c1) {
      func.call @__device_stub__g(%arg0, %arg1, %arg2, %arg3) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      gpu.terminator
    }
    return
  }
  func.func @host2(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i8) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0_i8 = arith.constant 0 : i8
    %0 = arith.cmpi ne, %arg4, %c0_i8 : i8
    scf.if %0 {
      gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c20, %arg15 = %c1, %arg16 = %c1) {
        func.call @__device_stub__f(%arg0, %arg1, %arg3) : (memref<?xf32>, memref<?xf32>, i32) -> ()
        gpu.terminator
      }
    } else {
      %1 = llvm.mlir.addressof @str1 : !llvm.ptr
      %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %3 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c20, %arg15 = %c1, %arg16 = %c1) {
      func.call @__device_stub__g(%arg0, %arg1, %arg2, %arg3) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      gpu.terminator
    }
    return
  }
  func.func @host3(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i8) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0_i8 = arith.constant 0 : i8
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c20, %arg15 = %c1, %arg16 = %c1) {
      func.call @__device_stub__f(%arg0, %arg1, %arg3) : (memref<?xf32>, memref<?xf32>, i32) -> ()
      gpu.terminator
    }
    %0 = arith.cmpi ne, %arg4, %c0_i8 : i8
    scf.if %0 {
      gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c20, %arg15 = %c1, %arg16 = %c1) {
        func.call @__device_stub__g(%arg0, %arg1, %arg2, %arg3) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
        gpu.terminator
      }
    } else {
      %1 = llvm.mlir.addressof @str1 : !llvm.ptr
      %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %3 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    return
  }
  func.func @host4(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i8, %arg5: i8) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0_i8 = arith.constant 0 : i8
    %0 = arith.cmpi ne, %arg4, %c0_i8 : i8
    scf.if %0 {
      gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c20, %arg16 = %c1, %arg17 = %c1) {
        func.call @__device_stub__f(%arg0, %arg1, %arg3) : (memref<?xf32>, memref<?xf32>, i32) -> ()
        gpu.terminator
      }
    } else {
      %2 = llvm.mlir.addressof @str1 : !llvm.ptr
      %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %4 = llvm.call @printf(%3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c20, %arg16 = %c1, %arg17 = %c1) {
      func.call @__device_stub__g(%arg0, %arg1, %arg2, %arg3) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      gpu.terminator
    }
    %1 = arith.cmpi ne, %arg5, %c0_i8 : i8
    scf.if %1 {
      gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c20, %arg16 = %c1, %arg17 = %c1) {
        func.call @__device_stub__h(%arg0, %arg1, %arg3) : (memref<?xf32>, memref<?xf32>, i32) -> ()
        gpu.terminator
      }
    } else {
      %2 = llvm.mlir.addressof @str1 : !llvm.ptr
      %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %4 = llvm.call @printf(%3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    return
  }
  func.func @host5(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i8, %arg5: i8) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0_i8 = arith.constant 0 : i8
    %0 = arith.cmpi ne, %arg4, %c0_i8 : i8
    scf.if %0 {
      gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c20, %arg16 = %c1, %arg17 = %c1) {
        func.call @__device_stub__h(%arg0, %arg1, %arg3) : (memref<?xf32>, memref<?xf32>, i32) -> ()
        gpu.terminator
      }
    } else {
      %2 = llvm.mlir.addressof @str1 : !llvm.ptr
      %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %4 = llvm.call @printf(%3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c20, %arg16 = %c1, %arg17 = %c1) {
      func.call @__device_stub__g(%arg0, %arg1, %arg2, %arg3) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      gpu.terminator
    }
    %1 = arith.cmpi ne, %arg5, %c0_i8 : i8
    scf.if %1 {
      gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c20, %arg16 = %c1, %arg17 = %c1) {
        func.call @__device_stub__f(%arg0, %arg1, %arg3) : (memref<?xf32>, memref<?xf32>, i32) -> ()
        gpu.terminator
      }
    } else {
      %2 = llvm.mlir.addressof @str1 : !llvm.ptr
      %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %4 = llvm.call @printf(%3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    return
  }
}
*/
