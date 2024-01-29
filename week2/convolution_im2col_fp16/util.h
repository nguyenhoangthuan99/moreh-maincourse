#pragma once
#include <cuda_fp16.h>

double get_time();

half *alloc_tensor(int N, int C, int H, int W);

void rand_tensor(half *m, int N, int C, int H, int W);

void zero_tensor(half *m, int N, int C, int H, int W);

void print_tensor(half *m, int N, int C, int H, int W);

void check_convolution(half *I, half *F, half *O, int N, int C, int H, int W,
                       int K, int R, int S, int pad_h, int pad_w, int stride_h,
                       int stride_w, int dilation_h, int dilation_w);