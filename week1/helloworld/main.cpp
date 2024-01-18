#include <cstdio>
#include <mpi.h>
#include <math.h>
#include <chrono>
// int main()
// {
//     MPI_Init(NULL, NULL);
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     char hostname[MPI_MAX_PROCESSOR_NAME];
//     int hostnamelen;
//     MPI_Get_processor_name(hostname, &hostnamelen);
//     printf("[%s] Hello, I am rank %d of size %d world!\n", hostname, rank, size);
//     MPI_Finalize();
//     return 0;
// }

// int main(int argc, char **argv)
// {
//     int process_Rank, size_Of_Cluster, message_Item;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
//     MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);
//     char hostname[MPI_MAX_PROCESSOR_NAME];
//     int hostnamelen;
//     MPI_Get_processor_name(hostname, &hostnamelen);

//     int count = 100000000;
//     double *buf =
//         (double *)malloc(count * sizeof(double));
//     int num_run = 11;
//     double total_time = 0;
//     double total_time_send = 0;
//     for (int i = 0; i < num_run; i++)
//     {
//         if (process_Rank == 0)
//         {
//             double start = MPI_Wtime();
//             MPI_Send(buf, count, MPI_DOUBLE, 1, 1234, MPI_COMM_WORLD);
//             double end = MPI_Wtime();
//             printf("[%s] Sent %d double\n", hostname, count);
//             printf("[%s] Bandwidth Send %lf Gb/s\n", hostname, ( count * sizeof(double) / 1e9) / (end-start));
//             if (i > 0)
//                 total_time_send += (end - start);
//         }
//         if (process_Rank == 1)
//         {
//             MPI_Status status;
//             double start = MPI_Wtime();
//             MPI_Recv(buf, count, MPI_DOUBLE, 0, 1234, MPI_COMM_WORLD, &status);
//             double end = MPI_Wtime();
//             printf("[%s] Received %d double in %lf seconds\n", hostname, count, end - start);
//             printf("[%s] Bandwidth Receive %lf Gb/s\n", hostname, (count * sizeof(double) / 1e9) / (end-start));
//             if (i > 0)
//                 total_time += (end - start);
//         }
//     }
//     printf("----------------------------------------\n");
//     printf("[%s] Bandwidth Receive %lf Gb/s\n", hostname, ((num_run-1) * count * sizeof(double) / 1e9) / total_time);
//     printf("[%s] Bandwidth Send %lf Gb/s\n", hostname, ((num_run-1) * count * sizeof(double) / 1e9) / total_time_send);
//     MPI_Finalize();
//     return 0;
// }

int main(int argc, char *argv[])
{
    int done = 0, n, myid, numprocs, i, rc;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x, a;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    while (!done)
    {
        if (myid == 0)
        {
            printf("Enter the number of intervals: (0 quits) ");
            scanf("%d", &n);
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n == 0)
            break;
        h = 1.0 / (double)n;
        sum = 0.0;
        for (i = myid + 1; i <= n; i += numprocs)
        {
            x = h * ((double)i - 0.5);
            sum += 4.0 / (1.0 + x * x);
        }
        mypi = h * sum;
        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        if (myid == 0)
        {
            printf("pi is approximately %.16f,Error is %.16f\n ", pi, fabs(pi - PI25DT));
        }
    }
    MPI_Finalize();
    return 0;
}
