#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <mpi.h>
#define CHECK_MPI(e)                                    \
  if ((e) != MPI_SUCCESS) {                             \
    printf("FATAL ERROR: %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                            \
  }

#define MB (1ull << 20)
static int mpi_rank;
static int mpi_world_size;
static int root = 0;
static int niter = 8;
static int buf_size_mb = 0;

static int *sendbuf;
static int *recvbuf;

static void test_Reduce(void) {
  CHECK_MPI(MPI_Reduce(sendbuf, recvbuf, buf_size_mb * MB, MPI_BYTE, MPI_SUM, root, MPI_COMM_WORLD));
}

static void test_Allreduce(void) {
  // TODO: FILL_IN_HERE
}

static void test_Bcast(void) {
  // TODO: FILL_IN_HERE
}

static void test_Scatter(void) {
  CHECK_MPI(MPI_Scatter(sendbuf, (buf_size_mb * MB) / mpi_world_size, MPI_BYTE,
        recvbuf, (buf_size_mb * MB) / mpi_world_size, MPI_BYTE,
        root, MPI_COMM_WORLD));
}

static void test_Gather(void) {
  // TODO: FILL_IN_HERE
}

static void test_Allgather(void) {
  // TODO: FILL_IN_HERE
}

static void test_Alltoall(void) {
  // TODO: FILL_IN_HERE
}

static void test_Reduce_scatter(void) {
  // TODO: FILL_IN_HERE
}

static void test_Scan(void) {
  // TODO: FILL_IN_HERE
}


static void print_help(const char *prog_name) {
  if (mpi_rank == 0) {
    printf("Usage: %s [collectives] buffer_size (MB, maximum)\n", prog_name);
    printf("Options:\n");
    printf("\t--reduce: test REDUCE \n");
    printf("\t--allreduce: test ALLREDUCE \n");
    printf("\t--bcast: test BCAST \n");
    printf("\t--scatter: test SCATTER \n");
    printf("\t--gather: test GATHER \n");
    printf("\t--allgather: test ALLGATHER \n");
    printf("\t--alltoall: test ALLTOALL \n");
    printf("\t--reduce-scatter: test REDUCE_SCATTER \n");
    printf("\t--scan: test SCAN \n");
  }
}

struct col_opt {
  const char *name;
  int do_test;
  void (*test_function)(void);
};

struct col_opt col_opts[] = {
  { "Reduce", 0, test_Reduce },
  { "Allreduce", 0, test_Allreduce },
  { "Bcast", 0, test_Bcast },
  { "Scatter", 0, test_Scatter },
  { "Gather", 0, test_Gather },
  { "Allgather", 0, test_Allgather },
  { "Alltoall", 0, test_Alltoall },
  { "Reduce_scatter", 0, test_Reduce_scatter },
  { "Scan", 0, test_Scan }
};

static void parse_opt(int argc, char **argv) {
  int c;
  struct option long_options[] = {
    { "reduce", no_argument, 0, 0 },
    { "allreduce", no_argument, 0, 0 },
    { "bcast", no_argument, 0, 0 },
    { "scatter", no_argument, 0, 0 },
    { "gather", no_argument, 0, 0 },
    { "allgather", no_argument, 0, 0 },
    { "alltoall", no_argument, 0, 0 },
    { "reduce-scatter", no_argument, 0, 0 },
    { "scan", no_argument, 0, 0 },
    { 0, 0, 0, 0 }
  };

  int option_index = 0;
  while ((c = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
    switch (c) {
      case 0:
        col_opts[option_index].do_test = 1;
        break;
      case 'h':
      default:
        print_help(argv[0]);
        MPI_Finalize();
        exit(0);
    }
  }

  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: buf_size_mb = atoll(argv[i]); break;
      default: break;
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  parse_opt(argc, argv);

  if (mpi_rank == 0) {
    printf("Buffer size: %d MB\n", buf_size_mb);
    printf("Targets:\n");
    for (int i = 0; i < sizeof(col_opts)/sizeof(struct col_opt); ++i) {
      if (col_opts[i].do_test) {
        printf("\t%s\n", col_opts[i].name);
      }
    }
  }

  sendbuf = malloc(buf_size_mb * MB);
  recvbuf = malloc(buf_size_mb * MB);

  if (!sendbuf || !recvbuf) {
    printf("Cannot allocate buffers\n");
    return 1;
  }

  for (int i = 0; i < buf_size_mb * MB / sizeof(int); ++i) {
    sendbuf[i] = i;
    recvbuf[i] = 0;
  }

  for (int i = 0; i < sizeof(col_opts)/sizeof(struct col_opt); ++i) {
    if (col_opts[i].do_test) {
      // Warm-up
      col_opts[i].test_function();

      double starttime, endtime;

      CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
      if (mpi_rank == 0) {
        printf("Testing %s...\n", col_opts[i].name);
        starttime = MPI_Wtime();
      }
      for (int j = 0; j < niter; ++j) {
        col_opts[i].test_function();
      }
      CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
      if (mpi_rank == 0) {
        endtime = MPI_Wtime();
        printf("\tAverage elapsed time: %lf seconds\n", (endtime-starttime) / niter);
      }
    }
  }

  MPI_Finalize();

  return 0;
}
