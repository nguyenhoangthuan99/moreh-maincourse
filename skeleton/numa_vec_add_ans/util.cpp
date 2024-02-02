#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <numa.h>
#include <sched.h>
#include <omp.h>
#include <algorithm>
#include <numaif.h>
#include <unistd.h>
#include <vector>
#include <errno.h>

static double start_time[8];

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) {
    start_time[i] = get_time();
}

double timer_stop(int i) {
    return get_time() - start_time[i];
}

void check_vec_add(float *A, float *B, float *C, int M) {
  printf("Validating...\n");

  float *C_ans;
  alloc_vec(&C_ans, M);
  zero_vec(C_ans, M);
  for (int i = 0; i < M; ++i) {
    C_ans[i] = A[i] + B[i];
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    float c = C[i];
    float c_ans = C_ans[i];
    if (fabsf(c - c_ans) > eps && (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("C[%d] : correct_value = %f, your_value = %f\n", i, c_ans, c);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

void print_vec(float *m, int R) {
  for (int i = 0; i < R; ++i) { 
    printf("%+.3f ", m[i]);
  }
  printf("\n");
}

void alloc_vec(float **m, int R) {
  *m = (float *) aligned_alloc(4096, sizeof(float) * R);
  if (*m == NULL) {
    printf("Failed to allocate memory for vector.\n");
    exit(0);
  }
}

void rand_vec(float *m, int R, int num_threads) {
  int pagesize = getpagesize();
  for (int tid = 0; tid < num_threads; ++tid) {
    std::vector<void*> pages;
    std::vector<int> nodes;
    std::vector<int> status;
    int is = R / num_threads * tid + std::min(tid, R % num_threads);
    int ie = R / num_threads * (tid + 1) + std::min(tid + 1, R % num_threads);
    int numa_node = (tid / 8) % 4;

    for (int i = is; i < ie; ++i) {
      m[i] = i;
      if ((size_t)&m[i] % pagesize == 0) {
        pages.push_back(&m[i]);
        nodes.push_back(numa_node);
        status.push_back(0);
      }
    }
    int ret = move_pages(0, pages.size(), pages.data(), nodes.data(), status.data(), MPOL_MF_MOVE);
    if (ret < 0) exit(1);
  }
}

void zero_vec(float *m, int R) {
  memset(m, 0, sizeof(float) * R);
}
