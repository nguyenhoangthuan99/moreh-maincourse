#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <immintrin.h>
#include <sched.h>
#include <math.h>
#include <omp.h>
#include <mpi-ext.h>
#include <numa.h>
#include <iostream>
#include "util.h"


// static float *A, *B, *C;
// static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;

#define ITILESIZE (64)
#define JTILESIZE (512)
#define KTILESIZE (480)

typedef struct
{
  float *_A;
  float *_B;
  float *_C;
  int _M;
  int _N;
  int _K;
  int _mpi_rank;
} ThreadParams;
void *mat_mul_thread(void *params)
{
  ThreadParams *threadParams = (ThreadParams *)params;
  float *A = threadParams->_A, *B = threadParams->_B, *C = threadParams->_C;
  int M = threadParams->_M, N = threadParams->_N, K = threadParams->_K;
  int kk, ii, jj;
  // if (threadParams->_mpi_rank>0)
  zero_mat(C, M, N);
  for (kk = 0; kk < K; kk += KTILESIZE)
    for (ii = 0; ii < M; ii += ITILESIZE)
      for (jj = 0; jj < N; jj += JTILESIZE)
      {
        int k, i, j;
        int limk = std::min(kk + KTILESIZE, K);
        int limi = std::min(ii + ITILESIZE, M);
        int limj = std::min(jj + JTILESIZE, N);
        for (k = kk; k < limk - 5; k += 6)
          for (i = ii; i < limi; i++)
          {
            __m256 a0 = _mm256_set1_ps(A[i * K + k + 0]);
            __m256 a1 = _mm256_set1_ps(A[i * K + k + 1]);
            __m256 a2 = _mm256_set1_ps(A[i * K + k + 2]);
            __m256 a3 = _mm256_set1_ps(A[i * K + k + 3]);
            __m256 a4 = _mm256_set1_ps(A[i * K + k + 4]);
            __m256 a5 = _mm256_set1_ps(A[i * K + k + 5]);
            for (j = jj; j < limj - 7; j += 8)
            {
              __m256 b0 = _mm256_loadu_ps(B + (k + 0) * N + j);
              __m256 b1 = _mm256_loadu_ps(B + (k + 1) * N + j);
              __m256 b2 = _mm256_loadu_ps(B + (k + 2) * N + j);
              __m256 b3 = _mm256_loadu_ps(B + (k + 3) * N + j);
              __m256 b4 = _mm256_loadu_ps(B + (k + 4) * N + j);
              __m256 b5 = _mm256_loadu_ps(B + (k + 5) * N + j);
              __m256 c = _mm256_loadu_ps(C + i * N + j);
              c = _mm256_fmadd_ps(a0, b0, c);
              c = _mm256_fmadd_ps(a1, b1, c);
              c = _mm256_fmadd_ps(a2, b2, c);
              c = _mm256_fmadd_ps(a3, b3, c);
              c = _mm256_fmadd_ps(a4, b4, c);
              c = _mm256_fmadd_ps(a5, b5, c);
              // _mm256_storeu_ps(C + i * N + j, c);
              C[i * N + j + 7] = c[7];
              C[i * N + j + 6] = c[6];
              C[i * N + j + 0] = c[0];
              C[i * N + j + 1] = c[1];
              C[i * N + j + 2] = c[2];
              C[i * N + j + 3] = c[3];
              C[i * N + j + 4] = c[4];
              C[i * N + j + 5] = c[5];
            }
            
            for (; j < limj; j++)
            {
              C[i * N + j] += A[i * K + (k + 0)] * B[(k + 0) * N + j];
              C[i * N + j] += A[i * K + (k + 1)] * B[(k + 1) * N + j];
              C[i * N + j] += A[i * K + (k + 2)] * B[(k + 2) * N + j];
              C[i * N + j] += A[i * K + (k + 3)] * B[(k + 3) * N + j];
              C[i * N + j] += A[i * K + (k + 4)] * B[(k + 4) * N + j];
              C[i * N + j] += A[i * K + (k + 5)] * B[(k + 5) * N + j];
            }
          }
        for (; k < limk; k++)
          for (i = ii; i < limi; i++)
          // #pragma omp simd
            for (j = jj; j < limj; j++)
              C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
  pthread_exit(NULL);
}
static void mat_mul_pthread(float *_A, float *_B, float *_C, int _M, int _N, int _K, int _num_threads, int _mpi_rank)
{
  int M = _M, N = _N, K = _K;
  int chunk_size = static_cast<int>(std::ceil(static_cast<double>(M) / _num_threads));
  pthread_t threads[_num_threads];
  ThreadParams params[_num_threads];
  pthread_attr_t attr[_num_threads];
  cpu_set_t cpus[_num_threads];
  // Initialize and start each thread
  for (int i = 0; i < _num_threads; ++i)
  {
    int start = i * chunk_size;
    int end = (i == _num_threads - 1) ? _M : (i + 1) * chunk_size;
    params[i]._A = _A + start * K;
    params[i]._B = _B;
    params[i]._C = _C + start * N;
    params[i]._M = end - start;
    params[i]._N = _N;
    params[i]._K = _K;
    params[i]._mpi_rank = _mpi_rank;
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);
    pthread_attr_init(&attr[i]); // Initialize thread attributes
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&threads[i], &attr[i], mat_mul_thread, (void *)&params[i]);
  }
  // Wait for all threads to finish
  for (int i = 0; i < _num_threads; ++i)
    pthread_join(threads[i], NULL);
}

// static void mat_mul_omp(float *A, float *B, float *C, int M, int N, int K, int num_threads)
// {
//   // TODO: parallelize & optimize matrix multiplication
//   // Use num_threads per node
//   int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;
//   for (int kk = 0; kk < K; kk += T_k)
//   {
// #pragma omp parallel for simd num_threads(num_threads)

//     for (int ii = 0; ii < M; ii += T_i)
//     {
//       cpu_set_t mask;
//       CPU_ZERO(&mask);
//       int tid = omp_get_thread_num();
//       CPU_SET(tid, &mask);
//       sched_setaffinity(0, sizeof(mask), &mask);
//       for (int jj = 0; jj < N; jj += T_j)
//       {
//         int k, i, j;
//         int limk = std::min(K, kk + T_k);
//         int limi = std::min(M, ii + T_i);
//         int limj = std::min(N, jj + T_j);
//         for (k = kk; k < limk - 5; k += 6)
//         {
//           // #pragma simd

//           for (i = ii; i < limi; i++)
//           {
//             float a0 = A[i * K + k];
//             float a1 = A[i * K + k + 1];
//             float a2 = A[i * K + k + 2];
//             float a3 = A[i * K + k + 3];
//             float a4 = A[i * K + k + 4];
//             float a5 = A[i * K + k + 5];
//             // #pragma simd
//             for (j = jj; j < limj; j++)
//             {
//               float b0 = B[k * N + j];
//               float b1 = B[(k + 1) * N + j];
//               float b2 = B[(k + 2) * N + j];
//               float b3 = B[(k + 3) * N + j];
//               float b4 = B[(k + 4) * N + j];
//               float b5 = B[(k + 5) * N + j];

//               C[i * N + j] += (a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
//             }
//           }
//         }
//         // #pragma simd
//         for (; k < limk; k++)
//           for (i = ii; i < limi; i++)
//             for (j = jj; j < limj; j++)
//               C[i * N + j] += A[i * K + k] * B[k * N + j];
//       }
//     }
//   }
// }

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size)
//  float *A_scatter, float *B_bcast, float *C_scatter)
{
  float *A = _A;
  float *B = _B;
  float *C = _C;
  int M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;
  MPI_Request request[2];// = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
  int sendcounts[mpi_world_size] ;//= (int *)malloc(mpi_world_size * sizeof(int));
  int displ[mpi_world_size];// = (int *)malloc(mpi_world_size * sizeof(int));
  for (int i = 0; i < mpi_world_size; i++)
  {
    int is = M / mpi_world_size * i + std::min(i, M % mpi_world_size);
    int ie = M / mpi_world_size * (i+1) + std::min(i+1, M % mpi_world_size);
    sendcounts[i] = (ie-is)* K ;
    displ[i] = is * K ;
    // std::cout<<"\n"<<sendcounts[i]<<" "<<displ[i]<<std::endl;
  }

  MPI_Ibcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, request);

  MPI_Iscatterv(A, sendcounts,displ, MPI_FLOAT, A, M * K / mpi_world_size, MPI_FLOAT, 0, MPI_COMM_WORLD, request + 1);
  M = sendcounts[_mpi_rank]/K;
  MPI_Request request_result;
  // MPI_Status status_result;
  // MPI_Waitall(2,request,MPI_STATUSES_IGNORE);
  // double start = MPI_Wtime();
  mat_mul_pthread(A, B, C, M , N, K, num_threads, _mpi_rank);
  // double end = MPI_Wtime();

  // printf("\nexecution time %lf\n", end - start);
  // double start = MPI_Wtime();

  MPI_Igatherv(C, M * N , MPI_FLOAT, C, sendcounts,displ , MPI_FLOAT, 0, MPI_COMM_WORLD, &request_result);
  // MPI_Wait(&request_result, MPI_STATUS_IGNORE);
  // double end = MPI_Wtime();
  // printf("gather time %lf\n", end - start);
  // free(request);
  // free(sendcounts);
}
