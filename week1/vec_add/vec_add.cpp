#include "vec_add.h"

#include <pthread.h>
#include <immintrin.h>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <omp.h>
struct ThreadArg
{
  int start;
  int end;
  float *A;
  float *B;
  float *C;
};

void *add(void *arg)
{
  // std::cout<<"thread enter"<<std::endl;
  ThreadArg *myarg = (ThreadArg *)arg;
  // int tile = 16;
  // for (int ii = myarg->start; ii < myarg->end; ii += tile)
  // {
  for (int indx = myarg->start; indx < myarg->end; indx++)
  {
    myarg->C[indx] = myarg->A[indx] + myarg->B[indx];
  }
  // }

  // std::cout<<"thread exit"<<std::endl;
  return NULL;
}

void vec_add_pthread(float *_A, float *_B, float *_C, int _M, int _num_threads)
{
  // IMPLEMENT HERE
  pthread_t tid[_num_threads + 1];
  ThreadArg arg[_num_threads + 1];
  int _range = _M / _num_threads;

  for (int i = 0; i <= _num_threads; i++)
  {
    arg[i].start = i * _range;
    arg[i].end = std::min((i + 1) * _range, _M);
    arg[i].A = _A;
    arg[i].B = _B;
    arg[i].C = _C;

    pthread_create(&tid[i], NULL, add, &arg[i]);
  }
  // for (int i = _num_threads * _range; i < _M; i++)
  // {
  //   _C[i] = _A[i] + _B[i];
  // }

  for (int i = 0; i < _num_threads; ++i)
  {
    pthread_join(tid[i], NULL);
  }
}

void vec_add(float *_A, float *_B, float *_C, int _M, int _num_threads)
{
  float *A = _A;
  float *B = _B;
  float *C = _C;
  int M = _M;

#pragma omp parallel for simd num_threads(_num_threads) proc_bind(spread) shared(_A,_B,_C,_M)
  // #pragma omp simd
  for (int i = 0; i < M; ++i)
  {
    // cpu_set_t mask;
    // CPU_ZERO(&mask);
    // int tid = omp_get_thread_num();
    // CPU_SET(tid, &mask);
    // sched_setaffinity(0, sizeof(mask), &mask);
    _C[i] = _A[i] + _B[i];
  }
}