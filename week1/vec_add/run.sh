#!/bin/bash

srun --cpu-freq high --exclusive numactl --interleave=all ./main -n 10 -t 16 -v 10000000
