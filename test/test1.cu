#include "cuda.h"

#define N 20

extern "C"{
  __global__ void bar(double *w){
    w[threadIdx.x] = 2.0;
  }
}

extern "C" {
  void baz(double *w) {
    bar<<<1, N>>>(w);
  }
}
