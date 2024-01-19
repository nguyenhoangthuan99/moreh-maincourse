#!/bin/bash

salloc -N 2 --cpu-freq high --exclusive numactl --interleave=all  \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -v -t 32 -n 10 4096 4096 4096