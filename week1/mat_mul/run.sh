#!/bin/bash

srun --cpu-freq high --exclusive numactl --interleave=all ./main -v   -n 10 -t 32 5 5 5

# srun --cpu-freq high --exclusive numactl --interleave=all ./main  -n 10 -t 32 2048 2048 2048

# srun --cpu-freq high --exclusive numactl --interleave=all ./main -n 10 -t 32 4096 4096 4096
#  numactl --interleave=all