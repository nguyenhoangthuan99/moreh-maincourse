#include <cstdio>

#include "reduction.h"

#define THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void reduction_kernel(const int N, const float *a, const float *b, float *c) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  c[tidx] = a[tidx] + b[tidx];
}

// Device(GPU) pointers
static float *input_gpu, *output_gpu;

__global__ void reduce_kernel(float *input, float *output, int N) {
  extern __shared__ float L[];

  unsigned int tid = threadIdx.x;
  unsigned int offset = blockIdx.x * blockDim.x * 2;
  unsigned int stride = blockDim.x;

  L[tid] = 0;
  if (tid + offset < N) L[tid] += input[tid + offset];
  if (tid + stride + offset < N) L[tid] += input[tid + stride + offset];
  __syncthreads();

  for (stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) L[tid] += L[tid + stride];
    __syncthreads();
  }
  
  if (tid == 0) output[blockIdx.x] = L[0];
}


float reduction(float* A, float* B, int num_elements) {
  size_t output_elements = (num_elements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

  // (TODO) Upload A vector to GPU

  // Launch kernel on a GPU
  dim3 gridDim(output_elements);
  dim3 blockDim(THREADS_PER_BLOCK);
  reduce_kernel<<<gridDim, blockDim, THREADS_PER_BLOCK * sizeof(float), 0>>>
                             (input_gpu, output_gpu, num_elements);

  float sum = 0.0;
  // (TODO) Download B vector from GPU and sum it on CPU
  
  return sum;
}


void reduction_init(int N) {
  // (TODO) Allocate device memory

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void reduction_cleanup(float *_A, float *_B) {
  // (TODO) Do any post-reduction cleanup work here.

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}






