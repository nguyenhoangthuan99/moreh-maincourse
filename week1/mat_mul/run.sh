#!/bin/bash

srun --cpu-freq high --exclusive numactl --interleave=all ./main -n 10 -t 32 4096 4096 4096
