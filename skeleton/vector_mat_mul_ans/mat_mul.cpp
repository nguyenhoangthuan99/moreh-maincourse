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

  for (int k = 0; k < K; k += KTILESIZE) {
    for (int i = 0; i < M; i += ITILESIZE) {
      for (int j = 0; j < N; j += JTILESIZE) {

        for (int kk = k; kk < k + KTILESIZE; kk+=4) {
          for (int ii = i; ii < i + ITILESIZE; ii++) {
            __m256 a0 = _mm256_set1_ps(A[(ii+0)*K+(kk+0)]);
            __m256 a1 = _mm256_set1_ps(A[(ii+0)*K+(kk+1)]);
            __m256 a2 = _mm256_set1_ps(A[(ii+0)*K+(kk+2)]);
            __m256 a3 = _mm256_set1_ps(A[(ii+0)*K+(kk+3)]);
            //__m256 a4 = _mm256_set1_ps(A[(ii+0)*K+(kk+4)]);
            //__m256 a5 = _mm256_set1_ps(A[(ii+0)*K+(kk+5)]);
            //__m256 a6 = _mm256_set1_ps(A[(ii+0)*K+(kk+6)]);
            //__m256 a7 = _mm256_set1_ps(A[(ii+0)*K+(kk+7)]);

            for (int jj = j; jj < j + JTILESIZE; jj+=16) {
              __m256 c0 = _mm256_load_ps(&C[(ii+0) * N + jj]);


              __m256 b0 = _mm256_load_ps(&B[(kk+0) * N + jj]);
              __m256 b1 = _mm256_load_ps(&B[(kk+1) * N + jj]);
              __m256 b2 = _mm256_load_ps(&B[(kk+2) * N + jj]);
              __m256 b3 = _mm256_load_ps(&B[(kk+3) * N + jj]);
              //__m256 b4 = _mm256_load_ps(&B[(kk+4) * N + jj]);
              //__m256 b5 = _mm256_load_ps(&B[(kk+5) * N + jj]);
              //__m256 b6 = _mm256_load_ps(&B[(kk+6) * N + jj]);
              //__m256 b7 = _mm256_load_ps(&B[(kk+7) * N + jj]);

              c0 = _mm256_fmadd_ps(a0, b0, c0);
              c0 = _mm256_fmadd_ps(a1, b1, c0);
              c0 = _mm256_fmadd_ps(a2, b2, c0);
              c0 = _mm256_fmadd_ps(a3, b3, c0);
              //c0 = _mm256_fmadd_ps(a4, b4, c0);
              //c0 = _mm256_fmadd_ps(a5, b5, c0);
              //c0 = _mm256_fmadd_ps(a6, b6, c0);
              //c0 = _mm256_fmadd_ps(a7, b7, c0);

              __m256 d0 = _mm256_load_ps(&C[(ii+0) * N + jj+8]);

              __m256 e0 = _mm256_load_ps(&B[(kk+0) * N + jj+8]);
              __m256 e1 = _mm256_load_ps(&B[(kk+1) * N + jj+8]);
              __m256 e2 = _mm256_load_ps(&B[(kk+2) * N + jj+8]);
              __m256 e3 = _mm256_load_ps(&B[(kk+3) * N + jj+8]);
              //__m256 e4 = _mm256_load_ps(&B[(kk+4) * N + jj+8]);
              //__m256 e5 = _mm256_load_ps(&B[(kk+5) * N + jj+8]);
              //__m256 e6 = _mm256_load_ps(&B[(kk+6) * N + jj+8]);
              //__m256 e7 = _mm256_load_ps(&B[(kk+7) * N + jj+8]);

              d0 = _mm256_fmadd_ps(a0, e0, d0);
              d0 = _mm256_fmadd_ps(a1, e1, d0);
              d0 = _mm256_fmadd_ps(a2, e2, d0);
              d0 = _mm256_fmadd_ps(a3, e3, d0);
              //d0 = _mm256_fmadd_ps(a4, e4, d0);
              //d0 = _mm256_fmadd_ps(a5, e5, d0);
              //d0 = _mm256_fmadd_ps(a6, e6, d0);
              //d0 = _mm256_fmadd_ps(a7, e7, d0);

              _mm256_store_ps(&C[(ii+0)*N+jj], c0);
              _mm256_store_ps(&C[(ii+0)*N+jj+8], d0);
            }
          }
        }
      }
    }
  }
}
