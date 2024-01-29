#include <cstdio>
#include <math.h>

#include "vecadd.h"
#include "util.h"

int blocksize = 512;
#define CHECK_CUDA(call)                                                 \
  do                                                                     \
  {                                                                      \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess)                                          \
    {                                                                    \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void vecadd_kernel(const int N, const float *a, const float *b, float *c)
{
  
  __shared__ float A[512], B[512];
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  A[threadIdx.x] = a[tidx];
  B[threadIdx.x] = b[tidx];
  __syncthreads();
  if (tidx < N)
  {
    float a_ = A[threadIdx.x]; // a[tidx]; //
    float b_ = B[threadIdx.x]; // b[tidx]; //
    c[tidx] = a_ + b_;
  }
}

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void vecadd(float *_A, float *_B, float *_C, int N)
{
  // (TODO) Upload A and B vector to GPU
  // vecadd_init(N);
  CHECK_CUDA(cudaMemcpyAsync(A_gpu, _A, sizeof(float) * N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(B_gpu, _B, sizeof(float) * N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(C_gpu, _C, sizeof(float) * N, cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaDeviceSynchronize());
  // Launch kernel on a GPU
  dim3 blockDim(blocksize);
  dim3 gridDim(std::ceil(((float)N) / blockDim.x));
  // double start = get_time();
  vecadd_kernel<<<gridDim, blockDim>>>(N, A_gpu, B_gpu, C_gpu);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  // double end = get_time();
  // printf("throughput %f GFLOPS\n", ((float)N)/(end-start)/1024/1024/1024);
  // CHECK_CUDA(cudaDeviceSynchronize());
  // (TODO) Download C vector from GPU

  CHECK_CUDA(cudaMemcpy(_C, C_gpu, sizeof(float) * N, cudaMemcpyDeviceToHost));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_init(int N)
{
  // (TODO) Allocate device memory

  CHECK_CUDA(cudaMalloc(&A_gpu, sizeof(float) * N));
  CHECK_CUDA(cudaMalloc(&B_gpu, sizeof(float) * N));
  CHECK_CUDA(cudaMalloc(&C_gpu, sizeof(float) * N));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_cleanup(float *_A, float *_B, float *_C, int N)
{
  // (TODO) Do any post-vecadd cleanup work here.
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
