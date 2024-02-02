#include "vec_add.h"

#include <pthread.h>
#include <immintrin.h>
#include <cstdio>
#include <algorithm>

static float *A, *B, *C;
static int M;

void vec_add(float *_A, float *_B, float *_C, int _M, int _num_threads) {
  A = _A, B = _B, C = _C;
  M = _M;

  #pragma omp target map(to: A[0:M], B[0:M]) map(from: C[0:M])
  for (int i = 0; i < M; ++i) {
    C[i] = A[i] + B[i];
  }
}
