#include "mat_mul.h"

#include <pthread.h>
#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <math.h>
#include <omp.h>

// #define _GNU_SOURCE
#include <sched.h>

#define ITILESIZE (32)
#define JTILESIZE (1024)
#define KTILESIZE (1024)
struct ThreadArg
{
  int M, N, K;
  float *A, *B, *C;
};
void *mul_thread(void *arg)
{
  ThreadArg *myarg = (ThreadArg *)arg;
  int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;
  int i, j, k, ii, jj, kk;
  int M = myarg->M, N = myarg->N, K = myarg->K;
  float *A = myarg->A;
  float *B = myarg->B;
  float *C = myarg->C;

  for (kk = 0; kk < K; kk += T_k)
  {
    for (ii = 0; ii < M; ii += T_i)
    {
      for (jj = 0; jj < N; jj += T_j)
      {
        int limk = std::min(K, kk + T_k);
        int limi = std::min(M, ii + T_i);
        int limj = std::min(N, jj + T_j);
        for (k = kk; k < limk - 3; k += 4)
        {
          // #pragma omp simd
          for (i = ii; i < limi; i++)
          {
            float a0 = A[i * K + k];
            float a1 = A[i * K + k + 1];
            float a2 = A[i * K + k + 2];
            float a3 = A[i * K + k + 3];
            // float a4 = A[i * K + k + 4];

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
        for (; k < limk; k++)
          for (i = ii; i < limi; i++)
            for (j = jj; j < limj; j++)
              C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
  // pthread_exit(NULL);
  return NULL;
}
void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K, int _num_threads)
{
  float *A = _A, *B = _B, *C = _C;
  int M = _M, N = _N, K = _K;
  // int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;

  // IMPLEMENT HERE: TILE ONLY
  // for (int kk = 0; kk < K; kk += T_k)
  // {
  //   for (int ii = 0; ii < M; ii += T_i)
  //   {
  //     for (int jj = 0; jj < N; jj += T_j)
  //     {
  //       for (int k = kk; k < std::min(K, kk + T_k); k++)
  //       {
  //         for (int i = ii; i < std::min(M, ii + T_i); i++)
  //         {
  //           for (int j = jj; j < std::min(N, jj + T_j); j++)
  //           {
  //             C[i * N + j] += A[i * K + k] * B[k * N + j];
  //           }
  //         }
  //       }
  //     }
  //   }}

  // TILE AND UNROLLING
  // int k, i, j, ii, jj, kk;
  // int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;

  // for (kk = 0; kk < K; kk += T_k)
  // {
  //   for (ii = 0; ii < M; ii += T_i)
  //   {
  //     for (jj = 0; jj < N; jj += T_j)
  //     {
        
  //       int limk = std::min(K, kk + T_k);
  //       int limi = std::min(M, ii + T_i);
  //       int limj = std::min(N, jj + T_j);
  //       for (k = kk; k < limk-3; k += 4)
  //       {
  //         // #pragma omp simd
  //         for (i = ii; i < limi; i++)
  //         {
  //           float a0 = A[i * K + k]  ;
  //           float a1 = A[i * K + k + 1] ;
  //           float a2 =  A[i * K + k + 2]  ;
  //           float a3 =  A[i * K + k + 3]  ;
  //           // float a4 = A[i * K + k + 4];
  //           for (j = jj; j < limj; j++)
  //           {
  //             float b0 =  B[k * N + j] ;
  //             float b1 =   B[(k + 1) * N + j]  ;
  //             float b2 =   B[(k + 2) * N + j] ;
  //             float b3 = B[(k + 3) * N + j] ;
  //             // float b4 = B[(k + 4) * M + j];

  //             C[i * N + j] += (a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3);
  //           }
  //         }
  //       }
  //       for (; k < limk; k++)
  //         for (i = ii; i < limi; i++)
  //           for (j = jj; j < limj; j++)
  //             C[i * N + j] += A[i * K + k] * B[k * N + j];
  //     }
  //   }
  // }

  // Pthread
  // int chunk_size = static_cast<int>(std::ceil(static_cast<double>(M) / _num_threads));
  // pthread_t tid[_num_threads];
  // ThreadArg arg[_num_threads];
  // cpu_set_t cpus[_num_threads];
  // pthread_attr_t attr[_num_threads];

  // for (int i = 0; i < _num_threads; i++)
  // {
  //   int start = i * chunk_size;
  //   int end = (i == _num_threads - 1) ? _M : (i + 1) * chunk_size;
  //   pthread_attr_init(&attr[i]);
  //   // arg[i].start_i = i * _range;
  //   // arg[i].end_i = std::min((i + 1) * _range, _M);
  //   arg[i].A = A + start * K;
  //   arg[i].B = B;
  //   arg[i].C = C + start * N;
  //   arg[i].M = end - start;
  //   arg[i].N = N;
  //   arg[i].K = K;

  //   CPU_ZERO(&cpus[i]);
  //   CPU_SET(i, &cpus[i]);

  //   pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);

  //   pthread_create(&tid[i], &attr[i], mul_thread, &arg[i]);
  // }
  // for (int i = 0; i < _num_threads; ++i)
  // {
  //   pthread_join(tid[i], NULL);
  // }

  // OPEN MP

  int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;

  // int chunk_size = static_cast<int>(std::ceil(static_cast<double>(M) / _num_threads));
  // ThreadArg arg[_num_threads];
  // cpu_set_t cpus[_num_threads];
  

  for (int kk = 0; kk < K; kk += T_k)
  {
#pragma omp parallel for simd num_threads(_num_threads) shared(A,B)

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

  // OPEN MP GPU
  //   int k, i, j, ii, jj, kk;
  // float a0, a1, a2, a3, b0, b1, b2, b3;
  // int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;

  //   int chunk_size = static_cast<int>(std::ceil(static_cast<double>(M) / _num_threads));
  //   // pthread_t tid[_num_threads];
  //   ThreadArg arg[_num_threads];
  //   cpu_set_t cpus[_num_threads];
  // // pthread_attr_t attr[_num_threads];
  // #pragma omp target  teams distribute parallel for simd num_teams(_num_threads)

  // // #pragma omp for simd
  //     for (int i = 0; i < _num_threads; i++)
  //     {
  //       CPU_ZERO(&cpus[i]);
  //       CPU_SET(i, &cpus[i]);
  //       int start = i * chunk_size;
  //       int end = (i == _num_threads - 1) ? _M : (i + 1) * chunk_size;
  //       // pthread_attr_init(&attr[i]);
  //       arg[i].A = A + start * K;
  //       arg[i].B = B;
  //       arg[i].C = C + start * N;
  //       arg[i].M = end - start;
  //       arg[i].N = N;
  //       arg[i].K = K;
  //       mul_thread(&arg[i]);
  //     }
}

//     tile               order   unroll  perf
//  32, 1024, 2048        k,i,j   x       12
// int T_i = 32,T_j=1024,T_k = 1024;      15.8
// int T_i = 32,T_j=2048,T_k = 1024;      16.67
//  pthread 32 256 256