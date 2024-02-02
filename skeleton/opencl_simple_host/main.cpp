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
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;

  // 1. Get OpenCL platform
  // clGetPlatformIDs(..., &platform, ...);

  // 2. Get OpenCL device
  // clGetDeviceIDs(..., &device, ...);

  // 3. Create OpenCL context
  // context = clCreateContext(..., &err);

  // 4. Create OpenCL command queue
  // queue = clCreateCommandQueue(..., &err);

  // 5. Compile program from "kernel.cl"
  // program = ...;

  // 6. Extract kernel from compiled program
  // kernel = clCreateKernel(..., &err);

  // 7. Create buffers
  // a_d = clCreateBuffer(..., &err);
  // CHECK_OPENCL(err);
  // b_d = clCreateBuffer(..., &err);
  // CHECK_OPENCL(err);
  // c_d = clCreateBuffer(..., &err);
  // CHECK_OPENCL(err);

  // 8. Write to device
  // clEnqueueWriteBuffer(..., a_d, ..., A, ...);
  // clEnqueueWriteBuffer(..., b_d, ..., B, ...);

  // 9. Setup kernel arguments
  // clSetKernelArg(kernel, 0, ...);
  // clSetKernelArg(kernel, 1, ...);
  // clSetKernelArg(kernel, 2, ...);
  // clSetKernelArg(kernel, 3, ...);

  // 10. Setup global work size and local work size
  // By OpenCL spec, global work size should be MULTIPLE of local work size
  // size_t gws[1] = {...}, lws[1] = {...};

  // 11. Run kernel
  // clEnqueueNDRangeKernel(..., gws, lws, ...);

  // 12. Read from device
  // clEnqueueReadBuffer(..., c_d, ..., C, ...);

  // 13. Free resources
  //CHECK_OPENCL(clReleaseMemObject(a_d));
  //CHECK_OPENCL(clReleaseMemObject(b_d));
  //CHECK_OPENCL(clReleaseMemObject(c_d));
  //CHECK_OPENCL(clReleaseKernel(kernel));
  //CHECK_OPENCL(clReleaseProgram(program));
  //CHECK_OPENCL(clReleaseCommandQueue(queue));
  //CHECK_OPENCL(clReleaseContext(context));

  return 0;
}
