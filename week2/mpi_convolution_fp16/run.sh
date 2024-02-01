#!/bin/bash

# salloc -N 2 --exclusive  \
#   mpirun --bind-to none -mca btl ^openib -npernode 1 \
#   ./main -v -n 3 13 5 51 313 511 19 3 1 3 1 3
  
  
# salloc -N 2 --exclusive numactl --interleave=all  \
#   mpirun --bind-to none -mca btl ^openib -npernode 1 \
#   ./main -n 20 256 512 64 64 1024 3 3 1 1 1 1 1 1

# salloc -N 2 mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v
# salloc -N 2 --exclusive  mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -n 3 8 64 256 256 128 3 3 1 1 1 1 1
# salloc -N 2 mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 4 32 55 55 55 55 55 1 1 1 1 1
# salloc -N 2 mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 7 16 34 54 68 20 20 1 1 1 1 1
# salloc -N 2 mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 5 32 128 128 64 3 3 1 1 1 1 1
# salloc -N 2 mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 1 3 127 129 128 3 3 2 2 2 1 1
# salloc -N 2 --exclusive  mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 13 5 51 313 511 19 3 1 3 1 3
salloc -N 2 mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -n 10 128 512 64 64 1024 3 3 1 1 1 1 1 1
# salloc -N 2 mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 1000 7 51 71 37 23 17 1 1 1 1 1
# salloc -N 2 mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 1771 3 25 21 17 21 11 1 1 1 1 1