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

#define ITILESIZE (32)   // 64
#define JTILESIZE (512) // 512
#define KTILESIZE (512) // 480
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

  for (kk = 0; kk < K; kk += KTILESIZE)
    for (ii = 0; ii < M; ii += ITILESIZE)
      for (jj = 0; jj < N; jj += JTILESIZE)
      {
        int k, i, j;
        int boundk = std::min(kk + KTILESIZE, K);
        int boundi = std::min(ii + ITILESIZE, M);
        int boundj = std::min(jj + JTILESIZE, N);
        for (k = kk; k < boundk - 5; k += 6)
          for (i = ii; i < boundi; i++)
          {
            __m256 a0 = _mm256_set1_ps(A[i * K + k + 0]);
            __m256 a1 = _mm256_set1_ps(A[i * K + k + 1]);
            __m256 a2 = _mm256_set1_ps(A[i * K + k + 2]);
            __m256 a3 = _mm256_set1_ps(A[i * K + k + 3]);
            __m256 a4 = _mm256_set1_ps(A[i * K + k + 4]);
            __m256 a5 = _mm256_set1_ps(A[i * K + k + 5]);
            for (j = jj; j < boundj - 7; j += 8)
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
            for (; j < boundj; j++)
            {
              C[i * N + j] += A[i * K + (k + 0)] * B[(k + 0) * N + j];
              C[i * N + j] += A[i * K + (k + 1)] * B[(k + 1) * N + j];
              C[i * N + j] += A[i * K + (k + 2)] * B[(k + 2) * N + j];
              C[i * N + j] += A[i * K + (k + 3)] * B[(k + 3) * N + j];
              C[i * N + j] += A[i * K + (k + 4)] * B[(k + 4) * N + j];
              C[i * N + j] += A[i * K + (k + 5)] * B[(k + 5) * N + j];
            }
          }
        for (; k < boundk; k++)
          for (i = ii; i < boundi; i++)
            for (j = jj; j < boundj; j++)
              C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
  pthread_exit(NULL);
}
void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K, int _num_threads)
{
  float *A = _A, *B = _B, *C = _C;
  int M = _M, N = _N, K = _K;
  // int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;

  // IMPLEMENT HERE: TILE ONLY
  // int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;
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
  //       for (k = kk; k < limk - 5; k += 6)
  //       {
  //         // #pragma omp simd
  //         for (i = ii; i < limi; i++)
  //         {
  //           float a0 = A[i * K + k];
  //           float a1 = A[i * K + k + 1];
  //           float a2 = A[i * K + k + 2];
  //           float a3 = A[i * K + k + 3];
  //           float a4 = A[i * K + k + 4];
  //           float a5 = A[i * K + k + 5];
  //           for (j = jj; j < limj; j++)
  //           {
  //             float b0 = B[k * N + j];
  //             float b1 = B[(k + 1) * N + j];
  //             float b2 = B[(k + 2) * N + j];
  //             float b3 = B[(k + 3) * N + j];
  //             float b4 = B[(k + 4) * M + j];
  //             float b5 = B[(k + 5) * M + j];

  //             C[i * N + j] += (a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a5 * b5 + a4 * b4);
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

  // SIMD
  //  int k, i, j, ii, jj, kk;
  //  int T_i = ITILESIZE, T_j = JTILESIZE, T_k = KTILESIZE;
  //  for (kk = 0; kk < K; kk += KTILESIZE)
  //  {
  //    for (ii = 0; ii < M; ii += ITILESIZE)
  //    {
  //      for (jj = 0; jj < N; jj += JTILESIZE)
  //      {
  //        int k, i, j;
  //        int boundk = std::min(kk + KTILESIZE, K);
  //        int boundi = std::min(ii + ITILESIZE, M);
  //        int boundj = std::min(jj + JTILESIZE, N);
  //        for (k = kk; k < boundk - 5; k += 6)
  //          for (i = ii; i < boundi; i++)
  //          {
  //            __m256 a0 = _mm256_set1_ps(A[i * K + k + 0]);
  //            __m256 a1 = _mm256_set1_ps(A[i * K + k + 1]);
  //            __m256 a2 = _mm256_set1_ps(A[i * K + k + 2]);
  //            __m256 a3 = _mm256_set1_ps(A[i * K + k + 3]);
  //            __m256 a4 = _mm256_set1_ps(A[i * K + k + 4]);
  //            __m256 a5 = _mm256_set1_ps(A[i * K + k + 5]);
  //            for (j = jj; j < boundj - 7; j += 8)
  //            {
  //              __m256 b0 = _mm256_loadu_ps(B + (k + 0) * N + j);
  //              __m256 b1 = _mm256_loadu_ps(B + (k + 1) * N + j);
  //              __m256 b2 = _mm256_loadu_ps(B + (k + 2) * N + j);
  //              __m256 b3 = _mm256_loadu_ps(B + (k + 3) * N + j);
  //              __m256 b4 = _mm256_loadu_ps(B + (k + 4) * N + j);
  //              __m256 b5 = _mm256_loadu_ps(B + (k + 5) * N + j);
  //              __m256 c = _mm256_loadu_ps(C + i * N + j);
  //              c = _mm256_fmadd_ps(a0, b0, c);
  //              c = _mm256_fmadd_ps(a1, b1, c);
  //              c = _mm256_fmadd_ps(a2, b2, c);
  //              c = _mm256_fmadd_ps(a3, b3, c);
  //              c = _mm256_fmadd_ps(a4, b4, c);
  //              c = _mm256_fmadd_ps(a5, b5, c);
  //              // _mm256_storeu_ps(C + i * N + j, c);
  //              C[i * N + j + 7] = c[7];
  //              C[i * N + j + 6] = c[6];
  //              C[i * N + j + 0] = c[0];
  //              C[i * N + j + 1] = c[1];
  //              C[i * N + j + 2] = c[2];
  //              C[i * N + j + 3] = c[3];
  //              C[i * N + j + 4] = c[4];
  //              C[i * N + j + 5] = c[5];
  //            }
  //            for (; j < boundj; j++)
  //            {
  //              C[i * N + j] += A[i * K + (k + 0)] * B[(k + 0) * N + j];
  //              C[i * N + j] += A[i * K + (k + 1)] * B[(k + 1) * N + j];
  //              C[i * N + j] += A[i * K + (k + 2)] * B[(k + 2) * N + j];
  //              C[i * N + j] += A[i * K + (k + 3)] * B[(k + 3) * N + j];
  //              C[i * N + j] += A[i * K + (k + 4)] * B[(k + 4) * N + j];
  //              C[i * N + j] += A[i * K + (k + 5)] * B[(k + 5) * N + j];
  //            }
  //          }
  //        for (; k < boundk; k++)
  //          for (i = ii; i < boundi; i++)
  //            for (j = jj; j < boundj; j++)
  //              C[i * N + j] += A[i * K + k] * B[k * N + j];
  //      }
  //    }
  //  }

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
    int jj, kk;
    // int chunk_size = static_cast<int>(std::ceil(static_cast<double>(M) / _num_threads));
    // ThreadArg arg[_num_threads];
    // cpu_set_t cpus[_num_threads];

    for (int kk = 0; kk < K; kk += T_k)
    {
  #pragma omp parallel for num_threads(_num_threads) shared(A, B)

      for (int ii = 0; ii < M; ii += T_i)
      {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        int tid = omp_get_thread_num();
        CPU_SET(tid, &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
              for (jj = 0; jj < N; jj += JTILESIZE)
       {
         int k, i, j;
         int boundk = std::min(kk + KTILESIZE, K);
         int boundi = std::min(ii + ITILESIZE, M);
         int boundj = std::min(jj + JTILESIZE, N);
         for (k = kk; k < boundk - 5; k += 6)
           for (i = ii; i < boundi; i++)
           {
             __m256 a0 = _mm256_set1_ps(A[i * K + k + 0]);
             __m256 a1 = _mm256_set1_ps(A[i * K + k + 1]);
             __m256 a2 = _mm256_set1_ps(A[i * K + k + 2]);
             __m256 a3 = _mm256_set1_ps(A[i * K + k + 3]);
             __m256 a4 = _mm256_set1_ps(A[i * K + k + 4]);
             __m256 a5 = _mm256_set1_ps(A[i * K + k + 5]);
             for (j = jj; j < boundj - 7; j += 8)
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
             for (; j < boundj; j++)
             {
               C[i * N + j] += A[i * K + (k + 0)] * B[(k + 0) * N + j];
               C[i * N + j] += A[i * K + (k + 1)] * B[(k + 1) * N + j];
               C[i * N + j] += A[i * K + (k + 2)] * B[(k + 2) * N + j];
               C[i * N + j] += A[i * K + (k + 3)] * B[(k + 3) * N + j];
               C[i * N + j] += A[i * K + (k + 4)] * B[(k + 4) * N + j];
               C[i * N + j] += A[i * K + (k + 5)] * B[(k + 5) * N + j];
             }
           }
         for (; k < boundk; k++)
           for (i = ii; i < boundi; i++)
             for (j = jj; j < boundj; j++)
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