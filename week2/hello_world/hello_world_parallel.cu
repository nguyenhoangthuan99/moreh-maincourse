#include <cstdio>

__global__ void hello_world()
{
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  printf("Device(GPU) Grid (%d, %d): Thread (%d %d) tid (%d %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,tidx,tidy);
}

int main()
{
  dim3 blockDim(2, 3);
  dim3 gridDim(1, 4);
  hello_world<<<gridDim, blockDim>>>();
  cudaDeviceSynchronize();
  return 0;
}
