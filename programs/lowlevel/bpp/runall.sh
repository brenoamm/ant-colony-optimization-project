#!/bin/bash

#nvcc -arch=compute_50 -code=sm_50 -O3 -use_fast_math -o bin/aco_gpu_ref -Iinclude/ src/aco_v3_cuda.cu src/aco_v3_cuda_algorithm.cu

echo "run; iterations; problem; colony size; Fitness; Total Time "
for KNAP in 7 6 5 4 3 1
do
	for RUN in 10
	do
		bin/mknap_aco_gpu_ref $RUN 50 $KNAP
	done	
done
