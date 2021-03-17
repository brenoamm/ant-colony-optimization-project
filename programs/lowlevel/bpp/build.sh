#!/bin/bash

#SBATCH --export=NONE
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu2080 
#SBATCH --time=00:10:00
#SBATCH --job-name=aco_cuda_p1
#SBATCH --mail-type=ALL
#SBATCH --output /home/b/b_mene01/outputs/compile.out
#SBATCH --mail-user=b_mene01@uni-muenster.de
# environment variables

 # environment variables 
module load GCC/8.2.0-2.31.1 
module load CUDA/10.1.105 
module load gompi/2019a 
module load Boost/1.70.0 
module load CMake/3.15.3 

module list

cd /home/b/b_mene01/ls_pi-research_aco-bpp/BPP/LowLevelProgram/build/Release/

cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Release /home/b/b_mene01/ls_pi-research_aco-bpp/BPP/LowLevelProgram/source/

make

