#!/bin/bash

#SBATCH --export=NONE
#SBATCH --ntasks 1 
#SBATCH --nodes 1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=4G 
#SBATCH --partition=gpu2080 
#SBATCH --exclusive 
#SBATCH --time=00:10:00 
#SBATCH --job-name=aco_mknap_gpu2080_job0 
#SBATCH --mail-type=ALL 
#SBATCH --output /home/b/b_mene01/aco-multidimensional-knapsack-problem/outputs/compile_mknap_aco_gpu2080_problem0.out 
#SBATCH --mail-user=b_mene01@uni-muenster.de 
# environment variables 

module load fosscuda/2019a
module load Boost/1.70.0
module load CMake/3.15.3

module list

cd /home/b/b_mene01/aco-multidimensional-knapsack-problem/build/

rm -rf Release/

mkdir Release

cd Release/

pwd

cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Release /home/b/b_mene01/aco-multidimensional-knapsack-problem/source/

make

