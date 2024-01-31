#include <cstdlib>
#include <cstdio>
#include <cuda_fp16.h>
#include "convolution.cuh"
#include "util.h"
#include <thread>
#include <cublas_v2.h>
#include <mpi.h>
#include <mpi-ext.h>
#define BLOCKS 16
#define BLOCK_SIZE 16
static int ngpu;
static half *I_gpu[1024], *F_gpu[1024], *BUF1_gpu[1024];
static float *BUF2_gpu[1024], *O_gpu[1024];
static int N_gpu_start[1024], N_gpu_end[1024];
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

__global__ void im2col_kernel(half *_I, half *workspace, int N, int C, int H, int W,
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
          // half temp = ;
          // for (int on = 0; on < ON; ++on)
          // {

          workspace[((c * R * S) + (r * S) + s) * (ON * OH * OW) +
                    (on * OH * OW + oh * OW + ow)] =
              _I[on * C * H * W + c * H * W + h * W + w];
          // }
        }
  }
}

__global__ void matmul_gpu(half *_A, half *_B, float *_C, int M, int N, int K, int temp_N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int k;
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float A1[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float A2[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float A3[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float A4[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float A5[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B1[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B2[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B3[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B4[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B5[BLOCK_SIZE][BLOCK_SIZE];

  // float sum[8] = {0.0f};//, sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
  float sum[6][6] = {0};
  float zero_float = 0.;
  // float4 sum2 = make_float4(0, 0, 0, 0);
  int boundk = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int aid = 6 * idy * K + threadIdx.x;
  int bid = threadIdx.y * N + idx * 6;
  // printf("%d %d - %d %d\n", aid, bid, idx, idy);
  for (k = 0; k < boundk; k++)
  {
    As[threadIdx.y][threadIdx.x] = (idy * 6 < M) && k * BLOCK_SIZE + threadIdx.x < K ? __half2float(_A[aid]) : zero_float;
    A1[threadIdx.y][threadIdx.x] = (idy * 6 + 1 < M) && k * BLOCK_SIZE + threadIdx.x < K ? __half2float(_A[aid + K]) : zero_float;
    A2[threadIdx.y][threadIdx.x] = (idy * 6 + 2 < M) && k * BLOCK_SIZE + threadIdx.x < K ? __half2float(_A[aid + 2 * K]) : zero_float;
    A3[threadIdx.y][threadIdx.x] = (idy * 6 + 3 < M) && k * BLOCK_SIZE + threadIdx.x < K ? __half2float(_A[aid + 3 * K]) : zero_float;
    A4[threadIdx.y][threadIdx.x] = (idy * 6 + 4 < M) && k * BLOCK_SIZE + threadIdx.x < K ? __half2float(_A[aid + 4 * K]) : zero_float;
    A5[threadIdx.y][threadIdx.x] = (idy * 6 + 5 < M) && k * BLOCK_SIZE + threadIdx.x < K ? __half2float(_A[aid + 5 * K]) : zero_float;

    Bs[threadIdx.y][threadIdx.x] = (idx * 6 < N) && k * BLOCK_SIZE + threadIdx.y < K ? __half2float(_B[bid]) : zero_float;
    B1[threadIdx.y][threadIdx.x] = (idx * 6 + 1 < N) && k * BLOCK_SIZE + threadIdx.y < K ? __half2float(_B[bid + 1]) : zero_float;
    B2[threadIdx.y][threadIdx.x] = (idx * 6 + 2 < N) && k * BLOCK_SIZE + threadIdx.y < K ? __half2float(_B[bid + 2]) : zero_float;
    B3[threadIdx.y][threadIdx.x] = (idx * 6 + 3 < N) && k * BLOCK_SIZE + threadIdx.y < K ? __half2float(_B[bid + 3]) : zero_float;
    B4[threadIdx.y][threadIdx.x] = (idx * 6 + 4 < N) && k * BLOCK_SIZE + threadIdx.y < K ? __half2float(_B[bid + 4]) : zero_float;
    B5[threadIdx.y][threadIdx.x] = (idx * 6 + 5 < N) && k * BLOCK_SIZE + threadIdx.y < K ? __half2float(_B[bid + 5]) : zero_float;
    __syncthreads();

    // #pragma unroll 32
    for (int ex = 0; ex < BLOCK_SIZE; ex++)
    {
      float a0 = As[threadIdx.y][ex];
      float a1 = A1[threadIdx.y][ex];
      float a2 = A2[threadIdx.y][ex];
      float a3 = A3[threadIdx.y][ex];
      float a4 = A4[threadIdx.y][ex];
      float a5 = A5[threadIdx.y][ex];
      float b0 = Bs[ex][threadIdx.x];
      float b1 = B1[ex][threadIdx.x];
      float b2 = B2[ex][threadIdx.x];
      float b3 = B3[ex][threadIdx.x];
      float b4 = B4[ex][threadIdx.x];
      float b5 = B5[ex][threadIdx.x];

      sum[0][0] += a0 * b0;
      sum[0][1] += a0 * b1;
      sum[0][2] += a0 * b2;
      sum[0][3] += a0 * b3;
      sum[0][4] += a0 * b4;
      sum[0][5] += a0 * b5;

      sum[1][0] += a1 * b0;
      sum[1][1] += a1 * b1;
      sum[1][2] += a1 * b2;
      sum[1][3] += a1 * b3;
      sum[1][4] += a1 * b4;
      sum[1][5] += a1 * b5;

      sum[2][0] += a2 * b0;
      sum[2][1] += a2 * b1;
      sum[2][2] += a2 * b2;
      sum[2][3] += a2 * b3;
      sum[2][4] += a2 * b4;
      sum[2][5] += a2 * b5;

      sum[3][0] += a3 * b0;
      sum[3][1] += a3 * b1;
      sum[3][2] += a3 * b2;
      sum[3][3] += a3 * b3;
      sum[3][4] += a3 * b4;
      sum[3][5] += a3 * b5;

      sum[4][0] += a4 * b0;
      sum[4][1] += a4 * b1;
      sum[4][2] += a4 * b2;
      sum[4][3] += a4 * b3;
      sum[4][4] += a4 * b4;
      sum[4][5] += a4 * b5;

      sum[5][0] += a5 * b0;
      sum[5][1] += a5 * b1;
      sum[5][2] += a5 * b2;
      sum[5][3] += a5 * b3;
      sum[5][4] += a5 * b4;
      sum[5][5] += a5 * b5;
    }
    __syncthreads();
    aid += BLOCK_SIZE;
    bid += BLOCK_SIZE * N;
  }
  // float sum[4] = {sum0, sum1, sum2, sum3};

  int temp_K = M, temp_OHOW = N / temp_N;
  float left = min(6, N - idx * 6);
  float left_j = min(6, M - idy * 6);
#pragma unroll
  for (int j = 0; j < left_j; j++)
  {
#pragma unroll
    for (int i = 0; i < left; i++)
    {
      int c_index = idx * 6 + i + (idy * 6 + j) * N;
      int ohow = c_index % temp_OHOW;
      int n = c_index / temp_OHOW % temp_N;
      k = c_index / temp_OHOW / temp_N;
      int new_c_index = ((n * temp_K + k) * temp_OHOW + ohow);
      _C[new_c_index] = sum[j][i];
    }
  }
}

void convolution_thread(half *_I, half *_F, float *_O, half *_BUF1, float *_BUF2, int N,
                        int C, int H, int W, int K, int R, int S, int pad_h, int pad_w,
                        int stride_h, int stride_w, int dilation_h, int dilation_w, int gpu_id = 0)
{
  // Remove this line after you complete the convolution on GPU
  // naive_cpu_convolution_im2col(_I, _F, _O, _BUF1, _BUF2, N, C, H, W, K, R, S,
  //                              pad_h, pad_w, stride_h, stride_w, dilation_h,
  //                              dilation_w);
  half *I = _I, *F = _F, *BUF1 = _BUF1;
  float *O = _O, *BUF2 = _BUF2;
  cudaSetDevice(gpu_id);
  int num_stream_blocks = min(BLOCKS, N);
  cudaStream_t data_h2d_stream, data_d2h_stream, calc_im2col_stream, calc_matmul_stream;
  cudaStream_t streams[num_stream_blocks];
  cudaStreamCreate(&data_h2d_stream);
  // cudaStreamCreate(&data_d2h_stream);
  // cudaStreamCreate(&calc_im2col_stream);
  // cudaStreamCreate(&calc_matmul_stream);
  cudaEvent_t events_data[num_stream_blocks], events_im2col_cals[num_stream_blocks], events_matmul_cals[num_stream_blocks];

  for (int i = 0; i < num_stream_blocks; i++)
  {
    cudaEventCreate(&events_data[i]);
    cudaEventCreate(&events_im2col_cals[i]);
    cudaEventCreate(&events_matmul_cals[i]);
    cudaStreamCreate(&streams[i]);
  }

  int Nbegin[num_stream_blocks], Nend[num_stream_blocks];
  for (size_t i = 0; i < num_stream_blocks; i++)
  {
    Nbegin[i] = N / num_stream_blocks * i;
    Nend[i] = N / num_stream_blocks * (i + 1);
    if (i == num_stream_blocks - 1)
      Nend[i] = N;
  }

  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  cudaMemcpyAsync(F_gpu[gpu_id], _F, sizeof(half) * K * C * R * S, cudaMemcpyHostToDevice, data_h2d_stream); // data_h2d_stream

  for (int i = 0; i < num_stream_blocks; i++)
  {
    cudaMemcpyAsync(&I_gpu[gpu_id][Nbegin[i] * C * H * W], &_I[Nbegin[i] * C * H * W], sizeof(half) * (Nend[i] - Nbegin[i]) * C * H * W, cudaMemcpyHostToDevice, streams[i]);
    // cudaEventRecord(events_data[i], data_h2d_stream);
  }
  cudaStreamSynchronize(data_h2d_stream);
  for (int i = 0; i < num_stream_blocks; i++)
  {
    dim3 blockDimIm2Col(640);
    dim3 gridDimIm2Col((OH * OW * (Nend[i] - Nbegin[i]) + blockDimIm2Col.x - 1) / blockDimIm2Col.x);
    // cudaStreamWaitEvent(calc_im2col_stream, events_data[i]);
    im2col_kernel<<<gridDimIm2Col, blockDimIm2Col, 0, streams[i]>>>(&I_gpu[gpu_id][Nbegin[i] * C * H * W], &BUF1_gpu[gpu_id][Nbegin[i] * C * R * S * OH * OW],
                                                                    (Nend[i] - Nbegin[i]), C, H, W,
                                                                    R, S, pad_h, pad_w, stride_h,
                                                                    stride_w, dilation_h, dilation_w);
    // cudaEventRecord(events_im2col_cals[i], calc_im2col_stream);
    // CHECK_CUDA(cudaDeviceSynchronize());
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((((Nend[i] - Nbegin[i]) * OH * OW + 5) / 6 + blockDim.x - 1) / blockDim.x, ((K + 5) / 6 + blockDim.y - 1) / blockDim.y, 1);
    // cudaStreamWaitEvent(calc_matmul_stream, events_im2col_cals[i]);
    matmul_gpu<<<gridDim, blockDim, 0, streams[i]>>>(F_gpu[gpu_id], &BUF1_gpu[gpu_id][Nbegin[i] * C * R * S * OH * OW],
                                                     &BUF2_gpu[gpu_id][Nbegin[i] * K * OH * OW],
                                                     K, (Nend[i] - Nbegin[i]) * OH * OW, C * R * S,
                                                     (Nend[i] - Nbegin[i]));
    // start reshape GPU
    // dim3 reshape_dimBlock(16, 16);
    // dim3 reshape_dimGrid(((Nend[i] - Nbegin[i]) + reshape_dimBlock.x - 1) / reshape_dimBlock.x, (K + reshape_dimBlock.y - 1) / reshape_dimBlock.y);
    // reshape_gpu<<<reshape_dimGrid, reshape_dimBlock, K * (Nend[i] - Nbegin[i]) * OH * OW * sizeof(float),calc_matmul_stream>>>(&BUF2_gpu[Nbegin[i] * K * OH * OW], &O_gpu[Nbegin[i] * K * OH * OW], (Nend[i] - Nbegin[i]), K, OH, OW);

    //
    // cudaEventRecord(events_matmul_cals[i], calc_matmul_stream);

    // cudaStreamWaitEvent(data_d2h_stream, events_matmul_cals[i]);
    // cudaMemcpyAsync(&_O[Nbegin[i] * K * OH * OW], &BUF2_gpu[gpu_id][Nbegin[i] * K * OH * OW], sizeof(float) * K * (Nend[i] - Nbegin[i]) * OH * OW, cudaMemcpyDeviceToHost, streams[i]);//data_d2h_stream
  }
  for (int i = 0; i < num_stream_blocks; i++)
  {
    cudaMemcpyAsync(&_O[Nbegin[i] * K * OH * OW], &BUF2_gpu[gpu_id][Nbegin[i] * K * OH * OW], sizeof(float) * K * (Nend[i] - Nbegin[i]) * OH * OW, cudaMemcpyDeviceToHost, streams[i]); // data_d2h_stream
  }
  // cudaMemcpyAsync(_O, BUF2_gpu[gpu_id], sizeof(float) * K * N * OH * OW, cudaMemcpyDeviceToHost);
  CHECK_CUDA(cudaDeviceSynchronize());
  // double start = get_time();
  // reshape(BUF2, O, N, K, OH, OW);
  // double end = get_time();
  // printf("\nreshape time: %lf\n", end - start);
  // CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_node(half *_I, half *_F, float *_O, half *_BUF1, float *_BUF2, int N,
                      int C, int H, int W, int K, int R, int S, int pad_h, int pad_w,
                      int stride_h, int stride_w, int dilation_h, int dilation_w)
{
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  std::thread threads[ngpu];
  for (int i = 0; i < ngpu; i++)
    threads[i] = std::thread(convolution_thread, &_I[N_gpu_start[i] * C * H * W], _F, &_O[N_gpu_start[i] * K * OH * OW],
                             BUF1_gpu[i],
                             BUF2_gpu[i],
                             (N_gpu_end[i] - N_gpu_start[i]), C, H, W, K, R, S, pad_h, pad_w,
                             stride_h, stride_w, dilation_h, dilation_w, i);
  /* Wait for all threads finish */

  for (int i = 0; i < ngpu; i++)
    threads[i].join();
}
void convolution(half *_I, half *_F, float *_O, half *_BUF1, float *_BUF2, int N,
                 int C, int H, int W, int K, int R, int S, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w, int mpi_rank, int mpi_world_size)
{
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  MPI_Request request[2];         // = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
  MPI_Status status[2];
  int sendcounts[mpi_world_size]; //= (int *)malloc(mpi_world_size * sizeof(int));
  int recvcounts[mpi_world_size];
  int displ[mpi_world_size]; // = (int *)malloc(mpi_world_size * sizeof(int));
  int displ_O[mpi_world_size];

  int Nbegin[mpi_world_size], Nend[mpi_world_size];
  for (size_t i = 0; i < mpi_world_size; i++)
  {
    Nbegin[i] = N / mpi_world_size * i;
    Nend[i] = N / mpi_world_size * (i + 1);
    if (i == mpi_world_size - 1)
      Nend[i] = N;

    sendcounts[i] = (Nend[i] - Nbegin[i]) * C * H * W * sizeof(half);
    recvcounts[i] = (Nend[i] - Nbegin[i]) * K * OH * OW ;
    displ[i] = Nbegin[i] * C * H * W * sizeof(half);
    displ_O[i] = Nbegin[i] * K * OH * OW ;
  }
  MPI_Ibcast((void *)_F, K * C * R * S * sizeof(half), MPI_BYTE, 0, MPI_COMM_WORLD, request);

  MPI_Iscatterv((void *)_I, sendcounts, displ, MPI_BYTE, (void *)_I, N * C * H * W * sizeof(half), MPI_BYTE, 0, MPI_COMM_WORLD, request + 1);
  N = sendcounts[mpi_rank] / (C * H * W)/ sizeof(half);

  MPI_Request request_result;
  MPI_Waitall(2, request, status);
  
  convolution_node(_I, _F, _O, _BUF1, _BUF2, N,
                   C, H, W, K, R, S, pad_h, pad_w,
                   stride_h, stride_w, dilation_h, dilation_w);
  MPI_Igatherv(_O, N * K * OH * OW , MPI_FLOAT, _O, recvcounts, displ_O, MPI_FLOAT, 0, MPI_COMM_WORLD, &request_result);
  MPI_Wait(&request_result, MPI_STATUS_IGNORE);
}
void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w)
{
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  cudaGetDeviceCount(&ngpu);
  // ngpu = 1;
  printf("\nNum GPUs: %d\n", ngpu);
  for (size_t i = 0; i < ngpu; i++)
  {
    N_gpu_start[i] = N / ngpu * i;
    N_gpu_end[i] = N / ngpu * (i + 1);
    if (i == ngpu - 1)
      N_gpu_end[i] = N;
  }

  for (int i = 0; i < ngpu; i++)
  {
    cudaSetDevice(i);
    CHECK_CUDA(cudaMalloc(&I_gpu[i], sizeof(half) * (N_gpu_end[i] - N_gpu_start[i]) * C * H * W));
    // CHECK_CUDA(cudaMalloc(&O_gpu, sizeof(float) * ON * OH * OW * OC));
    CHECK_CUDA(cudaMalloc(&F_gpu[i], sizeof(half) * K * C * R * S));
    CHECK_CUDA(cudaMalloc(&BUF1_gpu[i], sizeof(half) * C * R * S * (N_gpu_end[i] - N_gpu_start[i]) * OH * OW));
    CHECK_CUDA(cudaMalloc(&BUF2_gpu[i], sizeof(float) * K * (N_gpu_end[i] - N_gpu_start[i]) * OH * OW));
  }

  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(half *_I, half *_F, float *_O, int N, int C, int H,
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
// const half alpha = 1.0f, beta = 0.0f;
// cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N * OH * OW, K, C * R * S,
//              &alpha, BUF1_gpu, CUDA_R_32F, N * OH * OW, F_gpu, CUDA_R_32F, C * R * S,
//              &beta, BUF2_gpu, CUDA_R_32F, N * OH * OW, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);