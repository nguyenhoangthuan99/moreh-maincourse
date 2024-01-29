#include <cstdio>

#include "image_rotation.h"

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

// Device(GPU) pointers
static float *input_images_gpu, *output_images_gpu;

void rotate_image_naive(float *input_images, float *output_images, int W, int H,
                        float sin_theta, float cos_theta, int num_src_images)
{
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  // Rotate images
  for (int i = 0; i < num_src_images; i++)
  {
    for (int dest_x = 0; dest_x < W; dest_x++)
    {
      for (int dest_y = 0; dest_y < H; dest_y++)
      {
        float xOff = dest_x - x0;
        float yOff = dest_y - y0;
        int src_x = (int)(xOff * cos_theta + yOff * sin_theta + x0);
        int src_y = (int)(yOff * cos_theta - xOff * sin_theta + y0);
        if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H))
        {
          output_images[i * H * W + dest_y * W + dest_x] =
              input_images[i * H * W + src_y * W + src_x];
        }
        else
        {
          output_images[i * H * W + dest_y * W + dest_x] = 0.0f;
        }
      }
    }
  }
}

__global__ void rotate_image_device(float *input_images, float *output_images, int W, int H,
                                    float sin_theta, float cos_theta, int num_src_images)
{
  // int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  // int idx = tidx % W;
  // int idy = tidx / W % H;
  // int idz = tidx / W / H;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  // Rotate images
  float new_px =0.0f;
  float xOff = idx - x0;
  float yOff = idy - y0;
  int src_x = (int)(xOff * cos_theta + yOff * sin_theta + x0);
  int src_y = (int)(yOff * cos_theta - xOff * sin_theta + y0);
  if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H))
  {
    new_px = input_images[idz * H * W + src_y * W + src_x];
  }
  else
  {
    new_px = 0.0f;
  }
  output_images[idz * H * W + idy * W + idx] = new_px;
}

void rotate_image(float *input_images, float *output_images, int W, int H,
                  float sin_theta, float cos_theta, int num_src_images)
{
  // Remove this line after you complete the image rotation on GPU
  // rotate_image_naive(input_images, output_images, W, H, sin_theta, cos_theta,
  //                    num_src_images);

  // (TODO) Upload input images to GPU
  CHECK_CUDA(cudaMemcpy(input_images_gpu, input_images, sizeof(float) * W * H * num_src_images, cudaMemcpyHostToDevice));

  dim3 blockDim(16,16,1);
  dim3 gridDim(std::ceil(((float)W) / blockDim.x), std::ceil(((float)H) / blockDim.x),num_src_images);
  // dim3 gridDim((num_src_images * H * W + 1023) / 1024);
  // (TODO) Launch kernel on GPU
  rotate_image_device<<<gridDim, blockDim>>>(input_images_gpu, output_images_gpu, W, H, sin_theta, cos_theta, num_src_images);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  // (TODO) Download output images from GPU
  CHECK_CUDA(cudaMemcpyAsync(output_images, output_images_gpu, sizeof(float) * W * H * num_src_images, cudaMemcpyDeviceToHost));
  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void rotate_image_init(int image_width, int image_height, int num_src_images)
{
  // (TODO) Allocate device memory

  CHECK_CUDA(cudaMalloc(&input_images_gpu, sizeof(float) * image_height * image_width * num_src_images));
  CHECK_CUDA(cudaMalloc(&output_images_gpu, sizeof(float) * image_height * image_width * num_src_images));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void rotate_image_cleanup()
{
  // (TODO) Free device memory
  cudaFree(input_images_gpu);
  cudaFree(output_images_gpu);
  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
