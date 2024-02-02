#pragma once

float reduction(float *_A, float *_B, int N);
void reduction_init(int N);
void reduction_cleanup(float *_A, float *_B);
