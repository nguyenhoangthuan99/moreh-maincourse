#pragma once

double get_time();

void timer_start(int i);

double timer_stop(int i);

void check_vec_add(float *A, float *B, float *C, int M);

void print_vec(float *m, int R);

void alloc_vec(float **m, int R);

void rand_vec(float *m, int R, int num_threads);

void zero_vec(float *m, int R);
