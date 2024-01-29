#include <cstdio>

#include "convolution.h"
#define BLOCKS 16
#define BLOCK_SIZE 32
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
static float *I_gpu, *F_gpu, *O_gpu;

__global__ void convolution_gpu(float *I_gpu, float *F_gpu, float *O_gpu, int N, int C, int H,
                                int W, int K, int R, int S, int pad_h, int pad_w,
                                int stride_h, int stride_w, int dilation_h,
                                int dilation_w)
{
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  float sum = 0.0f;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int ow = tidx % OW;
  int oh = (tidx) / OW % OH;
  int ok = ((tidx) / OW) / OH % K;
  int on = (((tidx) / OW) / OH) / K;

  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < S; ++s)
  // for (int indx = 0; indx < C * R * S; indx++)
  {
    // int s = indx % S;
    // int r = indx/S%R;
    // int c = indx/(S*R);
    int n = on;
    int h = oh * stride_h - pad_h + r * dilation_h;
    int w = ow * stride_w - pad_w + s * dilation_w;
    int k = ok;
    if (h < 0 || h >= H || w < 0 || w >= W)
      continue;
    sum += I_gpu[((n * C + c) * H + h) * W + w] *
           F_gpu[((k * C + c) * R + r) * S + s];
  }

  O_gpu[((on * K + ok) * OH + oh) * OW + ow] = sum;
}

void convolution(float *_I, float *_F, float *_O, int N, int C, int H, int W,
                 int K, int R, int S, int pad_h, int pad_w, int stride_h,
                 int stride_w, int dilation_h, int dilation_w)
{
  // Remove this line after you complete the convolution on GPU
  // naive_cpu_convolution(_I, _F, _O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
  //                       stride_w, dilation_h, dilation_w);
  cudaStream_t data_h2d_stream, data_d2h_stream, calc_stream;
  cudaStreamCreate(&data_h2d_stream);
  cudaStreamCreate(&data_d2h_stream);
  cudaStreamCreate(&calc_stream);
  cudaEvent_t events_data[BLOCKS], events_cals[BLOCKS];
  for (int i = 0; i < BLOCKS; i++)
  {
    cudaEventCreate(&events_data[i]);
    cudaEventCreate(&events_cals[i]);
  }
  int Nbegin[BLOCKS], Nend[BLOCKS];
  for (size_t i = 0; i < BLOCKS; i++)
  {
    Nbegin[i] = N / BLOCKS * i;
    Nend[i] = N / BLOCKS * (i + 1);
    if (i == BLOCKS - 1)
      Nend[i] = N;
  }
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  cudaMemcpyAsync(F_gpu, _F, sizeof(float) * K * C * R * S, cudaMemcpyHostToDevice);

  // cudaMemcpyAsync(I_gpu, _I, sizeof(float) * N * C * H * W, cudaMemcpyHostToDevice);
  for (int i = 0; i < BLOCKS; i++)
  {
    cudaMemcpyAsync(&I_gpu[Nbegin[i] * C * H * W], &_I[Nbegin[i] * C * H * W],
                    (Nend[i] - Nbegin[i]) * C * H * W * sizeof(float),
                    cudaMemcpyHostToDevice, data_h2d_stream);
    cudaEventRecord(events_data[i], data_h2d_stream);
  }
  // dim3 blockDim(512);
  // dim3 gridDim((N * K * OH * OW + 511) / 512);
  // CHECK_CUDA(cudaDeviceSynchronize());
  // convolution_gpu<<<gridDim, blockDim>>>(I_gpu, F_gpu, O_gpu, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  // CHECK_CUDA(cudaGetLastError());
  // CHECK_CUDA(cudaMemcpyAsync(_O, O_gpu, sizeof(float) * ON * OC * OH * OW, cudaMemcpyDeviceToHost));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < BLOCKS; i++)
  {
    dim3 blockDim(512);
    dim3 gridDim(((Nend[i] - Nbegin[i]) * K * OH * OW + blockDim.x - 1) / blockDim.x);
    cudaStreamWaitEvent(calc_stream, events_data[i]);
    convolution_gpu<<<gridDim, blockDim, 0, calc_stream>>>(&I_gpu[Nbegin[i] * C * H * W], F_gpu, &O_gpu[Nbegin[i] * K * OH * OW], (Nend[i] - Nbegin[i]), C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    cudaEventRecord(events_cals[i], calc_stream);
    cudaStreamWaitEvent(data_d2h_stream, events_cals[i]);
    CHECK_CUDA(cudaMemcpyAsync(&_O[Nbegin[i] * K * OH * OW], &O_gpu[Nbegin[i] * K * OH * OW], sizeof(float) * (Nend[i] - Nbegin[i]) * K * OH * OW, cudaMemcpyDeviceToHost, data_d2h_stream));
  }

  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w)
{
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  CHECK_CUDA(cudaMalloc(&I_gpu, sizeof(float) * N * C * H * W));
  CHECK_CUDA(cudaMalloc(&O_gpu, sizeof(float) * ON * OH * OW * OC));
  CHECK_CUDA(cudaMalloc(&F_gpu, sizeof(float) * K * C * R * S));

  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w)
{
  cudaFree(I_gpu);
  cudaFree(F_gpu);
  cudaFree(O_gpu);
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}