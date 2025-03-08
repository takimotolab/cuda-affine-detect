#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N 10000000
#define MAX_ERR 1e-6

extern "C"{
  __global__ void vector_add(float *out, float *a, float *b, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(tid < n)
      out[tid] = a[tid] + b[tid];
  }
}

extern "C"{
  __global__ void vector_mad(float *out, float *a, float *b, float* c, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
      out[tid] = a[tid] * b[tid] + c[tid];
  }
}

extern "C"{
  int main(int argc, char* argv[]) {
    const int n = atoi(argv[1]);
    float *a1, *b1, *out1;
    float *a2, *b2, *out2;
    float *d_a1, *d_b1, *d_out1;
    float *d_a2, *d_b2, *d_out2/*, *d_c2*/;

    a1 = (float*)malloc(sizeof(float)*N);
    b1 = (float*)malloc(sizeof(float)*N);
    out1 = (float*)malloc(sizeof(float)*N);

    a2 = (float*)malloc(sizeof(float)*N);
    b2 = (float*)malloc(sizeof(float)*N);
    out2 = (float*)malloc(sizeof(float)*N);

    for(int i=0;i<N;i++) {
      a1[i] = 1.0f;
      b1[i] = 2.0f;
      out1[i] = 0.0f;
      a2[i] = 1.2f;
      b2[i] = 2.3f;
    }
  
    cudaMalloc((void**)&d_a1, sizeof(float)*N);
    cudaMemcpy(d_a1, a1, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_b1, sizeof(float)*N);
    cudaMemcpy(d_b1, b1, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_out1, sizeof(float)*N);
    cudaMemcpy(d_out1, out1, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_a2, sizeof(float)*N);
    cudaMemcpy(d_a2, a2, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_b2, sizeof(float)*N);
    cudaMemcpy(d_b2, b2, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_out2, sizeof(float)*N);
  
    int block_size = 256;
    int grid_size = ((N+block_size) / (block_size));

    //test for case 2 and 3

    //kernel 1 launch
    vector_add<<<dim3(grid_size,2,2),dim3(block_size,2,2)>>>(d_out1, d_a1, d_b1, N);

    if(n < 10){
      //kernel 2 launch
      vector_mad<<<dim3(grid_size,2,2),dim3(block_size,2,2)>>>(d_out2, d_a2, d_b2, d_out1, N);
    }

    //copy the result of kernel from device to host
    cudaMemcpy(out1, d_out1, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, d_out2, sizeof(float) * N, cudaMemcpyDeviceToHost);

    //printf("out1[0] = %f\n", out1[0]);
    //printf("out2[0] = %f\n", out2[0]);
    //for(int i=0;i<100;i++) printf("out2[%d] = %f\n", i, out2[i]);
  
    for(int i=0;i<N;i++){
      assert(fabs(out1[i] - a1[i] - b1[i]) < MAX_ERR);
      if(n < 10){
        assert(fabs(out2[i] - (a2[i] * b2[i] + out1[i])) < MAX_ERR);
      }
    }
  
    printf("PASSED\n");
  
  
    free(a1);
    free(b1);
    free(out1);
  
    free(a2);
    free(b2);
    free(out2);

    cudaFree(d_a1);
    cudaFree(d_b1);
    cudaFree(d_out1);

    cudaFree(d_a2);
    cudaFree(d_b2);
    cudaFree(d_out2);
    //cudaFree(d_c2);
  }
}
