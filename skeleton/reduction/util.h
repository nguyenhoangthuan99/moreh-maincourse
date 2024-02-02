#pragma once

double get_time();

void check_reduction(float *A, float result, int N);

void print_vec(float *m, int N);

float* alloc_vec(int N);

void rand_vec(float *m, int N);

void zero_vec(float *m, int N);
