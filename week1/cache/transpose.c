/*
 * trans.c - Matrix transpose B = A^T
 *
 * Each transpose function must have a prototype of the form:
 * void trans(int M, int N, int A[N][M], int B[M][N]);
 *
 * A transpose function is evaluated by counting the number of misses
 * on a 1KB direct mapped cache with a block size of 32 bytes.
 */
#include <stdio.h>
#include "cachelab.h"

int is_transpose(int M, int N, int A[N][M], int B[M][N]);

/*
 * transpose - This is the solution transpose function that you
 *     will be graded on for Part B of the assignment. Do not change
 *     the description string "Transpose submission", as the driver
 *     searches for that string to identify the transpose function to
 *     be graded.
 */
char transpose_desc[] = "Transpose submission";
void transpose(int M, int N, int A[N][M], int B[M][N])
{
    /* TODO: FILL IN HERE */
    
    // int num_blocks = 32; // 1kb/32bytes /block
    // int array_size = M*N; // elements
    // int elements_per_block = 8;
    // int total_element_in_cache = 256;
    // int divider_col = 4;
    int big_sub_line = M/32, big_sub_col = N/32;
    int divider_line = 8, divider_col = 256/(N/divider_line)/big_sub_col; //M/divider_line/(total_element_in_cache/elements_per_block);
    int num_line = M / divider_line;
    int num_col = N / divider_col;
    printf("%d %d\n", divider_col,divider_line);
    for (int i = 0; i <= num_col; i++)
    {
        for (int j = 0; j <= num_line; j++)
        {
            for (int l = 0; l < divider_col; l++)
            {
                for (int k = 0; k < divider_line; k++)
                {
                    int indx = divider_line * j + k;
                    int indx2 = divider_col * i + l;
                    if (indx < M && indx2 < N)
                        B[indx][indx2] = A[indx2][indx];
                }
            }
        }
    }

    // int num_line = 64;
    // int num_elements_per_line = 32 / sizeof(int);
    // for(int i = 0; i < N; i += num_line)
    //     for(int j = 0; j < M; j += num_elements_per_line){
    //         int i_bound = i + num_line > N ? N : i + num_line;
    //         int j_bound = j + num_elements_per_line > M ? M : j + num_elements_per_line;
    //         for (int ii = i; ii < i_bound; ii++)
    //             for(int jj = j; jj < j_bound; jj++)
    //                 B[jj][ii] = A[ii][jj];
    //     }

    //  for (int i = 0;i<M;i++){
    //     for (int j = 0;j<=num_line;j++){
    //         for(int k =0; k<divider;k++){
    //             int indx = divider*j+k ;
    //             if (indx>= N)
    //                 break;
    //             B[i][indx] = A[indx][i];
    //         }
    //     }
    //   }

    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < M; j++)
    //     {

    //         B[j][i] = A[i][j];
    //     }
    // }
}

/*
 * registerFunctions - This function registers your transpose
 *     functions with the driver.  At runtime, the driver will
 *     evaluate each of the registered functions and summarize their
 *     performance. This is a handy way to experiment with different
 *     transpose strategies.
 */
void registerFunctions()
{
    /* Register your solution function */
    registerTransFunction(transpose, transpose_desc);
}

/*
 * is_transpose - This helper function checks if B is the transpose of
 *     A. You can check the correctness of your transpose by calling
 *     it before returning from the transpose function.
 */
int is_transpose(int M, int N, int A[N][M], int B[M][N])
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; ++j)
        {
            if (A[i][j] != B[j][i])
            {
                return 0;
            }
        }
    }
    return 1;
}
