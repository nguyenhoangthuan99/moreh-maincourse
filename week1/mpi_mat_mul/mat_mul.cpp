#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <immintrin.h>
#include <sched.h>
#include <math.h>
#include <omp.h>

static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;

#define ITILESIZE (32)
#define JTILESIZE (512)
#define KTILESIZE (256)

static void mat_mul_omp()
{
  // TODO: parallelize & optimize matrix multiplication
  // Use num_threads per node
  int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;
  for (int kk = 0; kk < K; kk += T_k)
  {
#pragma omp parallel for simd num_threads(num_threads)

    for (int ii = 0; ii < M; ii += T_i)
    {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      int tid = omp_get_thread_num();
      CPU_SET(tid, &mask);
      sched_setaffinity(0, sizeof(mask), &mask);
      for (int jj = 0; jj < N; jj += T_j)
      {
        int k, i, j;
        int limk = std::min(K, kk + T_k);
        int limi = std::min(M, ii + T_i);
        int limj = std::min(N, jj + T_j);
        for (k = kk; k < limk - 3; k += 4)
        {
          // #pragma simd

          for (i = ii; i < limi; i++)
          {
            float a0 = A[i * K + k];
            float a1 = A[i * K + k + 1];
            float a2 = A[i * K + k + 2];
            float a3 = A[i * K + k + 3];
            // float a4 = A[i * K + k + 4];
            // #pragma simd
            for (j = jj; j < limj; j++)
            {
              float b0 = B[k * N + j];
              float b1 = B[(k + 1) * N + j];
              float b2 = B[(k + 2) * N + j];
              float b3 = B[(k + 3) * N + j];
              // float b4 = B[(k + 4) * M + j];

              C[i * N + j] += (a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3);
            }
          }
        }
        // #pragma simd
        for (; k < limk; k++)
          for (i = ii; i < limi; i++)
            for (j = jj; j < limj; j++)
              C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size)
{
  // A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;

  // TODO: parallelize & optimize matrix multiplication on multi-node
  // You must allocate & initialize A, B, C for non-root processes

  // FIXME: for now, only root process runs the matrix multiplication.
  // if (mpi_rank == 0)
  //   mat_mul_omp();

  // M = M / mpi_world_size;
  int chunk_size = static_cast<int>(std::ceil(static_cast<double>(_M) / mpi_world_size));
  int start = mpi_rank * chunk_size;
  int end = (mpi_rank == _num_threads - 1) ? _M : (mpi_rank + 1) * chunk_size;
  A = _A + start * K;
  B = _B;
  M = end - start;
  C = _C + start * N;
  mat_mul_omp();
}
