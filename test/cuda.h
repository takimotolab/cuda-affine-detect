#include <stddef.h>

#include "__clang_cuda_builtin_vars.h"

#include <cuda_runtime.h>

#ifdef __constant__
#  undef __constant__
#endif
#ifdef __device__
#  undef __device__
#endif
#ifdef __global__
#  undef __global__
#endif
#ifdef __host__
#  undef __host__
#endif
#ifdef __shared__
#  undef __shared__
#endif
#ifdef __managed__
#  undef __managed__
#endif
#ifdef __launch_bounds__
#  undef __launch_bounds__(...)
#endif

#if __HIP__ || __CUDA__
#  define __constant__ __attribute__((constant))
#  define __device__ __attribute__((device))
#  define __global__ __attribute__((global))
#  define __host__ __attribute__((host))
#  define __shared__ __attribute__((shared))
#  if __HIP__
#    define __managed__ __attribute__((managed))
#  endif
#  define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#else
#  define __constant__
#  define __device__
#  define __global__
#  define __host__
#  define __shared__
#  define __managed__
#  define __launch_bounds__(...)
#endif

#if __HIP__ || HIP_PLATFORM
typedef struct hipStream *hipStream_t;
typedef enum hipError {} hipError_t;
int hipConfigureCall(
  dim3 gridSize,
  dim3 blockSize,
  size_t sharedSize = 0,
  hipStream_t stream = 0
);
extern "C" hipError_t __hipPushCallConfiguration(
  dim3 gridSize,
  dim3 blockSize,
  size_t sharedSize = 0,
  hipStream_t stream = 0
);
#  ifndef HIP_API_PER_THREAD_DEFAULT_STREAM
extern "C" hipError_t hipLaunchKernel(
  const void *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem,
  hipStream_t stream
);
#  else
extern "C" hipError_t hipLaunchKernel_spt(
  const void *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem,
  hipStream_t stream
);
#  endif
#else
extern "C" int cudaConfigureCall(
  dim3 gridSize,
  dim3 blockSize,
  size_t sharedSize = 0,
  cudaStream_t stream = 0
);
extern "C" int __cudaPushCallConfiguration(
  dim3 gridSize, dim3 blockSize,
  size_t sharedSize = 0,
  cudaStream_t stream = 0
);
extern "C" cudaError_t cudaLaunchKernel(
  const void *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem,
  cudaStream_t stream
);
#endif

extern "C" __device__ int printf(const char*, ...);
