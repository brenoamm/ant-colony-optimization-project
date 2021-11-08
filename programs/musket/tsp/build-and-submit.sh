#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/n/n_herr03/musket-build/out/ && \
rm -rf -- /home/n/n_herr03/musket-build/build/benchmark && \
mkdir -p /home/n/n_herr03/musket-build/build/benchmark && \

# run cmake
cd /home/n/n_herr03/musket-build/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarkpalma -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make aco_iroulette_0 && \
cd ${source_folder} && \

sbatch job.sh
