#include <cstdio>

__global__ void welcome_kernel() {
  printf("(Device) Welcome to GETP!\n");
}

void welcome() {
  printf("(Host) Welcome to GETP!\n");
  welcome_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
