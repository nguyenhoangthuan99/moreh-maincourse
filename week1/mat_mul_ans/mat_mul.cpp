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

  // TILING ONLY

/*
  for (int ii = 0; ii < M; ii += ITILESIZE) {
    for (int jj = 0; jj < N; jj += JTILESIZE) {
      for (int kk = 0; kk < K; kk += KTILESIZE) {

        for (int k = kk; k < std::min(K, kk + KTILESIZE); k++) {
          for (int i = ii; i < std::min(M, ii + ITILESIZE); i++) {
            float ar = A[i * K + k];
            for (int j = jj; j < std::min(N, jj + JTILESIZE); j++) {
              C[i * N + j] += ar * B[k * N + j];
            }
          }
        }

      }
    }
  }
  */

  // TILING + LOOP UNROLLING
  for (int kk = 0; kk < K; kk += KTILESIZE) {
    for (int ii = 0; ii < M; ii += ITILESIZE) {
      for (int jj = 0; jj < N; jj += JTILESIZE) {

        for (int k = kk; k < std::min(K, kk + KTILESIZE); k+=4) {
          for (int i = ii; i < std::min(M, ii + ITILESIZE); i++) {
            float a0 = A[i * K + (k + 0)];
            float a1 = A[i * K + (k + 1)];
            float a2 = A[i * K + (k + 2)];
            float a3 = A[i * K + (k + 3)];
            for (int j = jj; j < std::min(N, jj + JTILESIZE); j+=1) {
              float b0 = B[(k + 0) * N + (j + 0)];
              float b1 = B[(k + 1) * N + (j + 0)];
              float b2 = B[(k + 2) * N + (j + 0)];
              float b3 = B[(k + 3) * N + (j + 0)];

              float c0 = C[i * N + (j + 0)];

              c0 += a0 * b0;
              c0 += a1 * b1;
              c0 += a2 * b2;
              c0 += a3 * b3;

              C[i * N + (j + 0)] = c0;
            }
          }
        }

      }
    }
  }
}
