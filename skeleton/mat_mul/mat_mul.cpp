#include "mat_mul.h"

#include <pthread.h>
#include <immintrin.h>
#include <algorithm>

#define ITILESIZE (32)
#define JTILESIZE (1024)
#define KTILESIZE (1024)

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K, int _num_threads) {
  float *A = _A, *B = _B, *C = _C;
  int M = _M, N = _N, K = _K;

  // IMPLEMENT HERE

}
