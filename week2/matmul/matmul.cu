#include <cstdio>
#include <thread>
#include "matmul.h"
#include "util.h"
// extern __device__ int g[N];
#define BLOCK_SIZE 16
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
    As[threadIdx.y][threadIdx.x] = (idy * 6 < M) && k * BLOCK_SIZE + threadIdx.x < K ? _A[aid] : zero_float;
    A1[threadIdx.y][threadIdx.x] = (idy * 6 + 1 < M) && k * BLOCK_SIZE + threadIdx.x < K ? _A[aid + K] : zero_float;
    A2[threadIdx.y][threadIdx.x] = (idy * 6 + 2 < M) && k * BLOCK_SIZE + threadIdx.x < K ? _A[aid + 2 * K] : zero_float;
    A3[threadIdx.y][threadIdx.x] = (idy * 6 + 3 < M) && k * BLOCK_SIZE + threadIdx.x < K ? _A[aid + 3 * K] : zero_float;
    A4[threadIdx.y][threadIdx.x] = (idy * 6 + 4 < M) && k * BLOCK_SIZE + threadIdx.x < K ? _A[aid + 4 * K] : zero_float;
    A5[threadIdx.y][threadIdx.x] = (idy * 6 + 5 < M) && k * BLOCK_SIZE + threadIdx.x < K ? _A[aid + 5 * K] : zero_float;

    Bs[threadIdx.y][threadIdx.x] = (idx * 6 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid] : zero_float;
    B1[threadIdx.y][threadIdx.x] = (idx * 6 + 1 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 1] : zero_float;
    B2[threadIdx.y][threadIdx.x] = (idx * 6 + 2 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 2] : zero_float;
    B3[threadIdx.y][threadIdx.x] = (idx * 6 + 3 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 3] : zero_float;
    B4[threadIdx.y][threadIdx.x] = (idx * 6 + 4 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 4] : zero_float;
    B5[threadIdx.y][threadIdx.x] = (idx * 6 + 5 < N) && k * BLOCK_SIZE + threadIdx.y < K ? _B[bid + 5] : zero_float;
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
  // float sum[4][4] = {{sum00, sum01, sum02, sum03}, {sum10, sum11, sum12, sum13}, {sum20, sum21, sum22, sum23}, {sum30, sum31, sum32, sum33}};
  // reinterpret_cast<float4 *>(_C)[idx * 8 + idy * N] = sum;
  // reinterpret_cast<float4 *>(_C)[idx * 8 + 4 + idy * N] = sum;
  float left = min(6, N - idx * 6);
  float left_j = min(6, M - idy * 6);
  #pragma unroll 
  for (int j = 0; j < left_j; j++)
  {
    #pragma unroll
    for (int i = 0; i < left; i++)
      _C[idx * 6 + i + (idy * 6 + j) * N] = sum[j][i];
  }
}

void matmul_thread(float *_A, float *_B, float *_C, int M, int N, int K, int gpu_id = 0)
{
  // Remove this line after you complete the matmul on GPU
  // naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU
  double total = 0.;
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
    dim3 gridDim(((N + 5) / 6 + blockDim.x - 1) / blockDim.x, (((Mend[i] - Mbegin[i]) + 5) / 6 + blockDim.y - 1) / blockDim.y, 1);
    // dim3 gridDim((N + blockDim.x - 1) / blockDim.x, ((Mend[i] - Mbegin[i]) + blockDim.y - 1) / blockDim.y, 1);

    // printf("\n%d %d\n", ((N + 3) / 4 + blockDim.x - 1) / blockDim.x, ((Mend[i] - Mbegin[i]) + blockDim.y - 1) / blockDim.y);
    cudaStreamWaitEvent(calc_stream, events_data[i]);
    // double start = get_time();
    matmul_gpu<<<gridDim, blockDim, 0, calc_stream>>>(&A_gpu[gpu_id][Mbegin[i] * K], B_gpu[gpu_id], &C_gpu[gpu_id][Mbegin[i] * N], (Mend[i] - Mbegin[i]), N, K);
    // CHECK_CUDA(cudaDeviceSynchronize());
    // double end = get_time();
    // total += end-start;
    cudaEventRecord(events_cals[i], calc_stream);
    cudaStreamWaitEvent(data_d2h_stream, events_cals[i]);
    CHECK_CUDA(cudaMemcpyAsync(&_C[Mbegin[i] * N], &C_gpu[gpu_id][Mbegin[i] * N], sizeof(float) * N * (Mend[i] - Mbegin[i]), cudaMemcpyDeviceToHost, data_d2h_stream));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
  // printf("\n Gflop only calculate: %lf \n",2.0 * M * N * K / total / 1e9);
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
