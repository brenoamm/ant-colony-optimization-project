#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/n/n_herr03/BPP/BPP/MusketProgram/out/ && \
rm -rf -- /home/n/n_herr03/BPP/BPP/MusketProgram/benchmark && \
mkdir -p /home/n/n_herr03/BPP/BPP/MusketProgram/benchmark && \

# run cmake
cd /home/n/n_herr03/BPP/BPP/MusketProgram/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarkpalma -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make BPP_0 && \
cd ${source_folder} && \

sbatch exec.sh
