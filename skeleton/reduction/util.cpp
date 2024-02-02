#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <cmath>
#include <sys/time.h>
#include <omp.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void check_reduction(float *A, float result, int N) {
  printf("Validating...\n");

  float sum = 0.0f;

#pragma omp parallel for num_threads(20) reduction (+:sum)
  for (int i = 0; i < N; ++i){
    sum += A[i];
  }
    

  bool is_valid = false;
  float eps = 1e-3;
  
   printf("GPU : %f, CPU : %f \n",result, sum);

  if(fabs(fabs(result) - fabs(sum)) < eps)
    is_valid = true;

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}
  
void print_vec(float *m, int N) {
  for (int i = 0; i < N; ++i) { 
    printf("%+.3f ", m[i]);
  }
  printf("\n");
}

float* alloc_vec(int N) {
  float *m = (float *) aligned_alloc(32, sizeof(float) * N);
  return m;
}

void rand_vec(float *m, int N) {
  for (int i = 0; i < N; i++) { 
    m[i] = (float) rand() / RAND_MAX - 0.5;
  }
}

void zero_vec(float *m, int N) {
  memset(m, 0, sizeof(float) * N);
}
