#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
  MPI_Init(NULL, NULL);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Hello, I am rank %d of size %d world!\n", rank, size);

  int niter = 10;
  int nelem = 100000000;
  size_t nbytes = sizeof(float) * nelem;
  float* buf = (float*)malloc(nbytes);
  double st, et;
  if (rank == 0) {
    for (int i = 0; i < niter; i++) {
      st = get_time();
      MPI_Send(buf, nelem, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
      et = get_time();
      printf("Send time: %f, BW=%f GB/s\n", et - st, nbytes / 1e9 / (et - st));
    }
  } else if (rank == 1) {
    for (int i = 0; i < niter; i++) {
      st = get_time();
      MPI_Recv(buf, nelem, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      et = get_time();
      printf("Recv time: %f, BW=%f GB/s\n", et - st, nbytes / 1e9 / (et - st));
    }
  }
  MPI_Finalize();
  return 0;
}
