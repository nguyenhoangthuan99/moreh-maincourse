#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_OPENCL(err)                                         \
  if (err != CL_SUCCESS) {                                        \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE);                                           \
  }

int main() {
  cl_int err;

  /*
You will use the following APIs:
clGetPlatformIDs
clGetPlatformInfo
clGetDeviceIDs
clGetDeviceInfo

[Example output]
Number of platforms: 1
platform: 0
- CL_PLATFORM_NAME: NVIDIA CUDA
- CL_PLATFORM_VENDOR: NVIDIA Corporation
Number of devices: 1
device: 0
- CL_DEVICE_TYPE: 4
- CL_DEVICE_NAME: NVIDIA GeForce RTX 3090
- CL_DEVICE_MAX_WORK_GROUP_SIZE: 1024
- CL_DEVICE_GLOBAL_MEM_SIZE: 25446907904
- CL_DEVICE_LOCAL_MEM_SIZE: 49152
- CL_DEVICE_MAX_MEM_ALLOC_SIZE: 6361726976
...
*/

  return 0;
}
