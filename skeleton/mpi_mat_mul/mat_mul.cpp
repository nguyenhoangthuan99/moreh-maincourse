#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;

static void mat_mul_omp() {
  // TODO: parallelize & optimize matrix multiplication
  // Use num_threads per node
  for (int k = 0; k < K; ++k) {
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size) {
  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;

  // TODO: parallelize & optimize matrix multiplication on multi-node
  // You must allocate & initialize A, B, C for non-root processes

  // FIXME: for now, only root process runs the matrix multiplication.
  if (mpi_rank == 0)
    mat_mul_omp();
}
