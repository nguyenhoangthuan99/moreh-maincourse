#!/bin/bash

# salloc -N 2 --exclusive  \
#   mpirun --bind-to none -mca btl ^openib -npernode 1 \
#   ./main -v -n 3 4 32 55 55 55 55 55 12 12 4 3 1
  
  
salloc -N 2 --exclusive numactl --interleave=all  \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -n 10 256 512 64 64 1024 3 3 1 1 1 1