#include <cstdio>
#include <thread>
#include "matmul.h"
#include "util.h"
// extern __device__ int g[N];
#define BLOCK_SIZE 32
#define BLOCKS 4
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
// extern __device__ void vecmul(float *_A, float *_B, int idx, int idy, int K, int N, float *sum);
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
static int ngpu;
static float *A_gpu[1024], *B_gpu[1024], *C_gpu[1024], *AT_gpu[1024];
static int M_gpu_start[1024], M_gpu_end[1024];

__global__ void cuda_transpose(float *in, float *out)
{
  __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];
  int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  tile[threadIdx.y][threadIdx.x] = in[y * gridDim.x * BLOCK_SIZE + x];
  __syncthreads();
  x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
  y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
  out[y * gridDim.y * BLOCK_SIZE + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void matmul_tiling(float *A, float *B, float *C, int M, int N, int K)
{
}

__global__ void matmul_gpu(float *_A, float *_B, float *_C, int M, int N, int K)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int k;
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B1[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B2[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B3[BLOCK_SIZE][BLOCK_SIZE];
  // __shared__ float B4[BLOCK_SIZE][BLOCK_SIZE];
  // __shared__ float B5[BLOCK_SIZE][BLOCK_SIZE];
  // __shared__ float B6[BLOCK_SIZE][BLOCK_SIZE];
  // __shared__ float B7[BLOCK_SIZE][BLOCK_SIZE];
  float c0;
  // float sum[8] = {0.0f};//, sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
  float4 sum = make_float4(0, 0, 0, 0);
  // float4 sum2 = make_float4(0, 0, 0, 0);
  int boundk = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int aid = idy * K + threadIdx.x;
  int bid = threadIdx.y * N + idx * 4;
  // printf("%d %d - %d %d\n", aid, bid, idx, idy);
  for (k = 0; k < boundk; k++)
  {
    As[threadIdx.y][threadIdx.x] = (idy < M) && k * BLOCK_SIZE + threadIdx.x < K ? _A[aid] : 0;
    Bs[threadIdx.y][threadIdx.x] = (idx * 4 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid] : 0;
    B1[threadIdx.y][threadIdx.x] = (idx * 4 + 1 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 1] : 0;
    B2[threadIdx.y][threadIdx.x] = (idx * 4 + 2 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 2] : 0;
    B3[threadIdx.y][threadIdx.x] = (idx * 4 + 3 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 3] : 0;
    // B4[threadIdx.y][threadIdx.x] = (idx * 8 + 4 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 4] : 0;
    // B5[threadIdx.y][threadIdx.x] = (idx * 8 + 5 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 5] : 0;
    // B6[threadIdx.y][threadIdx.x] = (idx * 8 + 6 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 6] : 0;
    // B7[threadIdx.y][threadIdx.x] = (idx * 8 + 7 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 7] : 0;
    __syncthreads();

    // #pragma unroll 32
    for (int e = 0; e < BLOCK_SIZE; e++)
    {
      float a = As[threadIdx.y][e];
      sum = make_float4(sum.x + a * Bs[e][threadIdx.x], sum.y + a * B1[e][threadIdx.x], sum.z + a * B2[e][threadIdx.x], sum.w + a * B3[e][threadIdx.x]);
      // sum2 = make_float4(sum.x + a * B4[e][threadIdx.x], sum.y + a * B5[e][threadIdx.x], sum.z + a * B6[e][threadIdx.x], sum.w + a * B7[e][threadIdx.x]);
      // sum[0] += a * Bs[e][threadIdx.x];
      // sum[1] += a * B1[e][threadIdx.x];
      // sum[2] += a * B2[e][threadIdx.x];
      // sum[3] += a * B3[e][threadIdx.x];
      // sum[4] += a * B4[e][threadIdx.x];
      // sum[5] += a * B5[e][threadIdx.x];
      // sum[6] += a * B6[e][threadIdx.x];
      // sum[7] += a * B7[e][threadIdx.x];
    }
    __syncthreads();
    aid += BLOCK_SIZE;
    bid += BLOCK_SIZE * N;
  }
  if (idy < M)
  {
    // reinterpret_cast<float4 *>(_C)[idx * 8 + idy * N] = sum;
    // reinterpret_cast<float4 *>(_C)[idx * 8 + 4 + idy * N] = sum;
        float left = min(4, N - idx * 4);
    #pragma unroll 4
        for (int i = 0; i < left; i++)
        {
          _C[idx * 4 + i + idy * N] = ((float*)(&sum))[i];
        }
  }
}

void matmul_thread(float *_A, float *_B, float *_C, int M, int N, int K, int gpu_id = 0)
{
  // Remove this line after you complete the matmul on GPU
  // naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU
  cudaSetDevice(gpu_id);

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

  int Mbegin[BLOCKS], Mend[BLOCKS];
  for (size_t i = 0; i < BLOCKS; i++)
  {
    Mbegin[i] = M / BLOCKS * i;
    Mend[i] = M / BLOCKS * (i + 1);
    if (i == BLOCKS - 1)
      Mend[i] = M;
  }

  cudaMemcpyAsync(B_gpu[gpu_id], _B, sizeof(float) * N * K, cudaMemcpyHostToDevice, data_h2d_stream);

  // cudaMemcpyAsync(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  for (int i = 0; i < BLOCKS; i++)
  {
    cudaMemcpyAsync(&A_gpu[gpu_id][Mbegin[i] * K], &_A[Mbegin[i] * K],
                    (Mend[i] - Mbegin[i]) * K * sizeof(float),
                    cudaMemcpyHostToDevice, data_h2d_stream);
    cudaEventRecord(events_data[i], data_h2d_stream);
  }

  // (TODO) Launch kernel on a GPU
  for (int i = 0; i < BLOCKS; i++)
  {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(((N + 3) / 4 + blockDim.x - 1) / blockDim.x, ((Mend[i] - Mbegin[i]) + blockDim.y - 1) / blockDim.y, 1);
    // dim3 gridDim((N + blockDim.x - 1) / blockDim.x, ((Mend[i] - Mbegin[i]) + blockDim.y - 1) / blockDim.y, 1);

    // printf("\n%d %d\n", ((N + 3) / 4 + blockDim.x - 1) / blockDim.x, ((Mend[i] - Mbegin[i]) + blockDim.y - 1) / blockDim.y);
    cudaStreamWaitEvent(calc_stream, events_data[i]);
    matmul_gpu<<<gridDim, blockDim, 0, calc_stream>>>(&A_gpu[gpu_id][Mbegin[i] * K], B_gpu[gpu_id], &C_gpu[gpu_id][Mbegin[i] * N], (Mend[i] - Mbegin[i]), N, K);
    cudaEventRecord(events_cals[i], calc_stream);
    cudaStreamWaitEvent(data_d2h_stream, events_cals[i]);
    CHECK_CUDA(cudaMemcpyAsync(&_C[Mbegin[i] * N], &C_gpu[gpu_id][Mbegin[i] * N], sizeof(float) * N * (Mend[i] - Mbegin[i]), cudaMemcpyDeviceToHost, data_d2h_stream));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
void matmul(float *_A, float *_B, float *_C, int M, int N, int K)
{
  std::thread threads[ngpu];
  for (int i = 0; i < ngpu; i++)
    threads[i] = std::thread(matmul_thread, &_A[M_gpu_start[i] * K], _B, &_C[M_gpu_start[i] * N], M_gpu_end[i] - M_gpu_start[i], N, K, i);
  /* Wait for all threads finish */

  for (int i = 0; i < ngpu; i++)
    threads[i].join();
}
void matmul_init(int M, int N, int K)
{
  // (TODO) Allocate device memory
  cudaGetDeviceCount(&ngpu);
  ngpu = 1;
  for (size_t i = 0; i < ngpu; i++)
  {
    M_gpu_start[i] = M / ngpu * i;
    M_gpu_end[i] = M / ngpu * (i + 1);
    if (i == ngpu - 1)
      M_gpu_end[i] = M;
  }
  for (int i = 0; i < ngpu; i++)
  {
    cudaSetDevice(i);
    CHECK_CUDA(cudaMalloc(&A_gpu[i], (M_gpu_end[i] - M_gpu_start[i]) * K * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&AT_gpu[i], (M_gpu_end[i] - M_gpu_start[i]) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_gpu[i], (M_gpu_end[i] - M_gpu_start[i]) * N * sizeof(float)));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K)
{
  // (TODO) Do any post-matmul cleanup work here.
  cudaFree(A_gpu);
  // cudaFree(AT_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
