#include <getopt.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include "mat_mul.h"
#include "util.h"

static bool print_matrix = false;
static bool validation = false;
static int M = 8, N = 8, K = 8;
static int num_threads = 1;
static int num_iterations = 1;
static int mpi_rank, mpi_world_size;

static void print_help(const char *prog_name)
{
  if (mpi_rank == 0)
  {
    printf("Usage: %s [-pvh] [-t num_threads] [-n num_iterations] M N K\n",
           prog_name);
    printf("Options:\n");
    printf("  -p : print matrix data. (default: off)\n");
    printf("  -v : validate matrix multiplication. (default: off)\n");
    printf("  -h : print this page.\n");
    printf("  -t : number of threads (default: 1)\n");
    printf("  -n : number of iterations (default: 1)\n");
    printf("   M : number of rows of matrix A and C. (default: 8)\n");
    printf("   N : number of columns of matrix B and C. (default: 8)\n");
    printf(
        "   K : number of columns of matrix A and rows of B. (default: 8)\n");
  }
}

static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "pvht:n:")) != -1)
  {
    switch (c)
    {
    case 'p':
      print_matrix = true;
      break;
    case 'v':
      validation = true;
      break;
    case 't':
      num_threads = atoi(optarg);
      break;
    case 'n':
      num_iterations = atoi(optarg);
      break;
    case 'h':
    default:
      print_help(argv[0]);
      MPI_Finalize();
      exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j)
  {
    switch (j)
    {
    case 0:
      M = atoi(argv[i]);
      break;
    case 1:
      N = atoi(argv[i]);
      break;
    case 2:
      K = atoi(argv[i]);
      break;
    default:
      break;
    }
  }
  if (mpi_rank == 0)
  {
    printf("Options:\n");
    printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
    printf("  Number of threads: %d\n", num_threads);
    printf("  Number of iterations: %d\n", num_iterations);
    printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
    printf("  Validation: %s\n", validation ? "on" : "off");
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  // MPI_Init(&argc, &argv);
  int i;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &i);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Hello world from processor %s, rank %d out of %d\n", processor_name,
         mpi_rank, mpi_world_size);

  parse_opt(argc, argv);
  int sendcounts[mpi_world_size]; //= (int *)malloc(mpi_world_size * sizeof(int));
  int recvcounts[mpi_world_size];
  int displ[mpi_world_size]; // = (int *)malloc(mpi_world_size * sizeof(int));
  int displ_C[mpi_world_size];
  for (int i = 0; i < mpi_world_size; i++)
  {
    int is = M / mpi_world_size * i + std::min(i, M % mpi_world_size);           // M / _mpi_world_size * i;
    int ie = M / mpi_world_size * (i + 1) + std::min(i + 1, M % mpi_world_size); // i + 1 == _mpi_world_size ? M : M / _mpi_world_size * (i + 1); // M / _mpi_world_size * (i + 1) + std::min(i + 1, M % _mpi_world_size);//
    sendcounts[i] = (ie - is) * K;
    recvcounts[i] = (ie - is) * N;
    displ[i] = is * K;
    displ_C[i] = is * N;
    // std::cout << "\n"
    //           << sendcounts[i] << " " << displ[i] << std::endl;
  }
  float *A, *B, *C;
  if (mpi_rank == 0)
  {
    printf("[rank %d] Initializing matrix...\n", mpi_rank);
    alloc_mat(&A, M, K);
    alloc_mat(&B, K, N);
    alloc_mat(&C, M, N);
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    printf("[rank %d] Initializing matrix done!\n", mpi_rank);
  }
  else
  {
    A = (float *)numa_alloc_interleaved(sendcounts[mpi_rank]* sizeof(float));
    B = (float *)numa_alloc_interleaved(K * N * sizeof(float));
    C = (float *)numa_alloc_interleaved(recvcounts[mpi_rank] * sizeof(float));
    zero_mat(C, sendcounts[mpi_rank]/K, N);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // if (mpi_rank == 0)
  // {
  //   B_bcast = B;
  // }

  double elapsed_time_sum = 0;
  for (int i = 0; i < num_iterations; ++i)
  {
    if (mpi_rank == 0)
    {
      printf("[rank %d] Calculating...(iter=%d) ", mpi_rank, i);
      fflush(stdout);
      zero_mat(C, M, N);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timer_start(0);
    // zero_mat(C_scatter, M / mpi_world_size, N);
    mat_mul(A, B, C, M, N, K, num_threads, mpi_rank, mpi_world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = timer_stop(0);

    if (mpi_rank == 0 )
    {
      printf("%f sec\n", elapsed_time);
      elapsed_time_sum += elapsed_time;
    }
  }

  if (mpi_rank == 0)
  {
    if (print_matrix)
    {
      printf("MATRIX A:\n");
      print_mat(A, M, K);
      printf("MATRIX B:\n");
      print_mat(B, K, N);
      printf("MATRIX C:\n");
      print_mat(C, M, N);
    }

    if (validation)
    {
      check_mat_mul(A, B, C, M, N, K);
    }

    double elapsed_time_avg = elapsed_time_sum / (num_iterations );
    printf("[rank %d] Avg. time: %f sec\n", mpi_rank, elapsed_time_avg);
    printf("[rank %d] Avg. throughput: %f GFLOPS\n", mpi_rank,
           2.0 * M * N * K / elapsed_time_avg / 1e9);
  }

  MPI_Finalize();

  return 0;
}
