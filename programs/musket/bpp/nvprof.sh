#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/BPP/CUDA/out/ && \
rm -rf -- ~/build/mnp/BPP/cuda && \
mkdir -p ~/build/mnp/BPP/cuda && \

# run cmake
cd ~/build/mnp/BPP/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make BPP_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
