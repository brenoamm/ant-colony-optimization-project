#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/aco_iroulette/CUDA/out/ && \
rm -rf -- ~/build/mnp/aco_iroulette/cuda && \
mkdir -p ~/build/mnp/aco_iroulette/cuda && \

# run cmake
cd ~/build/mnp/aco_iroulette/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make aco_iroulette_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
