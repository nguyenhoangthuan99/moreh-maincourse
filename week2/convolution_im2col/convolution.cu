#include <cstdlib>
#include <cstdio>
#include "convolution.cuh"
#include "util.h"

#include <cublas_v2.h>

#define BLOCKS 16
#define BLOCK_SIZE 32
static int ngpu;
static float *I_gpu, *F_gpu, *O_gpu, *BUF1_gpu, *BUF2_gpu;
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

__global__ void im2col_kernel(float *_I, float *workspace, int N, int C, int H, int W,
                              int R, int S, int pad_h, int pad_w, int stride_h,
                              int stride_w, int dilation_h, int dilation_w)
{
  const int ON = N;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  // int ow = tidx % OW;
  // int oh = (tidx) / OW % OH;
  // int on = tidx / OW / OH;

  const int ow = tidx % OW;
  const int oh = (tidx / OW) % OH;
  const int on = tidx / (OH * OW);
  if (ow < OW && oh < OH && on < N)
  {
#pragma unroll
    for (int c = 0; c < C; ++c)
      for (int r = 0; r < R; ++r)
        for (int s = 0; s < S; ++s)
        {
          const int h = oh * stride_h - pad_h + r * dilation_h;
          const int w = ow * stride_w - pad_w + s * dilation_w;
          if (h < 0 || h >= H || w < 0 || w >= W)
            continue;
          // float temp = ;
          // for (int on = 0; on < ON; ++on)
          // {

          workspace[((c * R * S) + (r * S) + s) * (ON * OH * OW) +
                    (on * OH * OW + oh * OW + ow)] =
              _I[on * C * H * W + c * H * W + h * W + w];
          // }
        }
  }
}

__global__ void matmul_gpu(float *_A, float *_B, float *_C, int M, int N, int K, int temp_N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int k;
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B1[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B2[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B3[BLOCK_SIZE][BLOCK_SIZE];
  float c0;
  float4 sum = make_float4(0, 0, 0, 0);
  // float4 sum2 = make_float4(0, 0, 0, 0);
  int boundk = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int aid = idy * K + threadIdx.x;
  int bid = threadIdx.y * N + idx * 4;
  for (k = 0; k < boundk; k++)
  {
    As[threadIdx.y][threadIdx.x] = (idy < M) && k * BLOCK_SIZE + threadIdx.x < K ? _A[aid] : 0;
    Bs[threadIdx.y][threadIdx.x] = (idx * 4 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid] : 0;
    B1[threadIdx.y][threadIdx.x] = (idx * 4 + 1 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 1] : 0;
    B2[threadIdx.y][threadIdx.x] = (idx * 4 + 2 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 2] : 0;
    B3[threadIdx.y][threadIdx.x] = (idx * 4 + 3 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 3] : 0;
    __syncthreads();

    // #pragma unroll 32
    for (int e = 0; e < BLOCK_SIZE; e++)
    {
      float a = As[threadIdx.y][e];
      sum = make_float4(sum.x + a * Bs[e][threadIdx.x], sum.y + a * B1[e][threadIdx.x], sum.z + a * B2[e][threadIdx.x], sum.w + a * B3[e][threadIdx.x]);
    }
    __syncthreads();
    aid += BLOCK_SIZE;
    bid += BLOCK_SIZE * N;
  }
  if (idy < M)
  {

    int temp_K = M, temp_OHOW = N / temp_N;
    float left = min(4, N - idx * 4);
#pragma unroll 4
    for (int i = 0; i < left; i++)
    {
      int c_index = idx * 4 + i + idy * N;
      int ohow = c_index % temp_OHOW;
      int n = c_index / temp_OHOW % temp_N;
      k = c_index / temp_OHOW / temp_N;
      int new_c_index = ((n * temp_K + k) * temp_OHOW + ohow);

      _C[new_c_index] = ((float *)(&sum))[i];
    }
  }
}
__global__ void reshape_gpu(float *_src, float *_dst, int N, int K, int OH, int OW)
{
  extern __shared__ float shared[];
  size_t chunk = OH * OW;

  const int on = blockDim.x * blockIdx.x + threadIdx.x;
  const int k = blockDim.y * blockIdx.y + threadIdx.y;

  if (on < N && k < K)
  {
    const int src_offset = on * K * chunk + k * chunk;
    const int dst_offset = on * K * chunk + k * chunk;

    for (int i = 0; i < chunk; ++i)
    {
      shared[threadIdx.x * blockDim.y + threadIdx.y + i] =
          _src[src_offset + i];
    }

    __syncthreads();

    for (int i = 0; i < chunk; ++i)
    {
      _dst[dst_offset + i] =
          shared[threadIdx.x * blockDim.y + threadIdx.y + i];
    }
  }
}
void reshape(float *_src, float *_dst, int N, int K, int OH, int OW)
{
  size_t chunk = OH * OW;
#pragma omp parallel for num_threads(32)
  for (int k = 0; k < K; ++k)
    for (int on = 0; on < N; ++on)
    {
      // for (int idx = 0; idx < K * N; idx++)
      {
        // int k = idx % K;
        // int on = idx / K;
        memcpy((void *)(_dst + ((on * K + k) * chunk)),
               (void *)(_src + ((k * N + on) * chunk)), chunk * sizeof(float));
      }
    }
}

void convolution(float *_I, float *_F, float *_O, float *_BUF1, float *_BUF2, int N,
                 int C, int H, int W, int K, int R, int S, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w)
{
  // Remove this line after you complete the convolution on GPU
  // naive_cpu_convolution_im2col(_I, _F, _O, _BUF1, _BUF2, N, C, H, W, K, R, S,
  //                              pad_h, pad_w, stride_h, stride_w, dilation_h,
  //                              dilation_w);
  float *I = _I, *F = _F, *O = _O, *BUF1 = _BUF1, *BUF2 = _BUF2;
  cudaSetDevice(0);

  cudaStream_t data_h2d_stream, data_d2h_stream, calc_im2col_stream, calc_matmul_stream;
  cudaStreamCreate(&data_h2d_stream);
  cudaStreamCreate(&data_d2h_stream);
  cudaStreamCreate(&calc_im2col_stream);
  cudaStreamCreate(&calc_matmul_stream);
  cudaEvent_t events_data[BLOCKS], events_im2col_cals[BLOCKS], events_matmul_cals[BLOCKS];

  for (int i = 0; i < BLOCKS; i++)
  {
    cudaEventCreate(&events_data[i]);
    cudaEventCreate(&events_im2col_cals[i]);
    cudaEventCreate(&events_matmul_cals[i]);
  }

  int Nbegin[BLOCKS], Nend[BLOCKS];
  for (size_t i = 0; i < BLOCKS; i++)
  {
    Nbegin[i] = N / BLOCKS * i;
    Nend[i] = N / BLOCKS * (i + 1);
    if (i == BLOCKS - 1)
      Nend[i] = N;
  }

  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  cudaMemcpyAsync(F_gpu, _F, sizeof(float) * K * C * R * S, cudaMemcpyHostToDevice, data_h2d_stream);

  for (int i = 0; i < BLOCKS; i++)
  {
    cudaMemcpyAsync(&I_gpu[Nbegin[i] * C * H * W], &_I[Nbegin[i] * C * H * W], sizeof(float) * (Nend[i] - Nbegin[i]) * C * H * W, cudaMemcpyHostToDevice, data_h2d_stream);
    cudaEventRecord(events_data[i], data_h2d_stream);
  }

  for (int i = 0; i < BLOCKS; i++)
  {
    dim3 blockDimIm2Col(768);
    dim3 gridDimIm2Col((OH * OW * (Nend[i] - Nbegin[i]) + blockDimIm2Col.x - 1) / blockDimIm2Col.x);
    cudaStreamWaitEvent(calc_im2col_stream, events_data[i]);
    im2col_kernel<<<gridDimIm2Col, blockDimIm2Col, 0, calc_im2col_stream>>>(&I_gpu[Nbegin[i] * C * H * W], &BUF1_gpu[Nbegin[i] * C * R * S * OH * OW],
                                                                            (Nend[i] - Nbegin[i]), C, H, W,
                                                                            R, S, pad_h, pad_w, stride_h,
                                                                            stride_w, dilation_h, dilation_w);
    cudaEventRecord(events_im2col_cals[i], calc_im2col_stream);
    // CHECK_CUDA(cudaDeviceSynchronize());
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((((Nend[i] - Nbegin[i]) * OH * OW + 3) / 4 + blockDim.x - 1) / blockDim.x, (K + blockDim.y - 1) / blockDim.y, 1);
    cudaStreamWaitEvent(calc_matmul_stream, events_im2col_cals[i]);
    matmul_gpu<<<gridDim, blockDim, 0, calc_matmul_stream>>>(F_gpu, &BUF1_gpu[Nbegin[i] * C * R * S * OH * OW],
                                                             &BUF2_gpu[Nbegin[i] * K * OH * OW],
                                                             K, (Nend[i] - Nbegin[i]) * OH * OW, C * R * S,
                                                             (Nend[i] - Nbegin[i]));
    cudaEventRecord(events_matmul_cals[i], calc_matmul_stream);

    cudaStreamWaitEvent(data_d2h_stream, events_matmul_cals[i]);
    cudaMemcpyAsync(&_O[Nbegin[i] * K * OH * OW], &BUF2_gpu[Nbegin[i] * K * OH * OW], sizeof(float) * K * (Nend[i] - Nbegin[i]) * OH * OW, cudaMemcpyDeviceToHost, data_d2h_stream);
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  // double start = get_time();
  // reshape(BUF2, O, N, K, OH, OW);
  // double end = get_time();
  // printf("\nreshape time: %lf\n", end - start);
  // CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_init(int N, int C, int H, int W, int K, int R, int S,
                      int pad_h, int pad_w, int stride_h, int stride_w,
                      int dilation_h, int dilation_w)
{

  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  CHECK_CUDA(cudaMalloc(&I_gpu, sizeof(float) * N * C * H * W));
  // CHECK_CUDA(cudaMalloc(&O_gpu, sizeof(float) * ON * OH * OW * OC));
  CHECK_CUDA(cudaMalloc(&F_gpu, sizeof(float) * K * C * R * S));
  CHECK_CUDA(cudaMalloc(&BUF1_gpu, sizeof(float) * C * R * S * ON * OH * OW));
  CHECK_CUDA(cudaMalloc(&BUF2_gpu, sizeof(float) * K * N * OH * OW));
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w)
{
  cudaFree(I_gpu);
  cudaFree(F_gpu);
  // cudaFree(O_gpu);
  cudaFree(BUF1_gpu);
  cudaFree(BUF2_gpu);
  CHECK_CUDA(cudaDeviceSynchronize());
}
// cublasHandle_t handle;
// cublasCreate(&handle);
// const float alpha = 1.0f, beta = 0.0f;
// cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N * OH * OW, K, C * R * S,
//              &alpha, BUF1_gpu, CUDA_R_32F, N * OH * OW, F_gpu, CUDA_R_32F, C * R * S,
//              &beta, BUF2_gpu, CUDA_R_32F, N * OH * OW, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);