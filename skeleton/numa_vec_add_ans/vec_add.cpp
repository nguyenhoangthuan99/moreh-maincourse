#include "vec_add.h"

#include <pthread.h>
#include <immintrin.h>
#include <cstdio>
#include <algorithm>
#include <omp.h>
#include <numa.h>
#include <numaif.h>
#include "util.h"

static float *A, *B, *C;
static int M;
static int num_threads;

// get numa node it is allocated on
int get_numa_node(float* buf) {
  int numa_node = -1;
  int ret = get_mempolicy(&numa_node, NULL, 0, buf, MPOL_F_NODE | MPOL_F_ADDR);
  if (ret < 0) {
    perror("get_mempolicy");
    exit(1);
  }
  return numa_node;
}

static void* vec_add_thread(void *data) {
  int tid = (long)data;
  int is = M / num_threads * tid + std::min(tid, M % num_threads);
  int ie = M / num_threads * (tid + 1) + std::min(tid + 1, M % num_threads);

  bitmask *cpumask = numa_allocate_cpumask();
  numa_bitmask_setbit(cpumask, tid);
  numa_sched_setaffinity(0, cpumask);
  numa_free_cpumask(cpumask);

  for (int i = is; i < ie; ++i) {
    C[i] = A[i] + B[i];
  }

  return NULL;
}


void vec_add(float *__restrict__ _A, float *__restrict__ _B, float *__restrict__ _C, int _M, int _num_threads) {
  A = _A, B = _B, C = _C;
  M = _M;
  num_threads = _num_threads;

  pthread_t threads[num_threads];
  pthread_attr_t attr[num_threads];
  cpu_set_t cpus[num_threads];
  for (long i = 0; i < num_threads; ++i) {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&threads[i], NULL, vec_add_thread, (void*)i);
  }
  for (long i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], NULL);
  }
}
