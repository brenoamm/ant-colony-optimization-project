#!/bin/bash

nvcc -arch=compute_50 -code=sm_50 -O3 -use_fast_math src/BPP_0.cu -o bin/BPP_0
echo "Ants;Problem;itemtypes;itemcount;Time;packingtimeperiteration;OptSolution;"
for BinSetup in 0 1 #2
do
    for Ant in 1024 2048 4096
    do
        for Run in 1
        do
            bin/BPP_0 5 $BinSetup $Ant
        done
    done
done
echo "/n"
