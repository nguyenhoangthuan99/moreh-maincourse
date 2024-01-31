#!/bin/bash

# salloc -N 2 --exclusive  \
#   mpirun --bind-to none -mca btl ^openib -npernode 1 \
#   ./main -v -n 3 17 5 51 677 511 19 3 2 3 2 3
  
  
salloc -N 2 --exclusive numactl --interleave=all  \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -n 10 256 512 64 64 1024 3 3 1 1 1 1