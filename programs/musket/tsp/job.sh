#!/bin/bash
#SBATCH --job-name ACO_Nina
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=32
#SBATCH --partition gpuk20
#SBATCH --exclusive
#SBATCH --error /home/n/n_herr03/musket-build/ACO.dat
#SBATCH --output /home/n/n_herr03/musket-build/ACO.dat
#SBATCH --mail-type ALL
#SBATCH --mail-user n_herr03@uni-muenster.de
#SBATCH --time 00:10:00
#SBATCH --gres=gpu:3

module load gcccuda/2019a  
module load CUDA/10.1.105

source_folder=/home/n/n_herr03/musket-build/ && \

mkdir -p /home/n/n_herr03/musket-build/out/ && \
rm -rf -- /home/n/n_herr03/musket-build/build/benchmark && \
mkdir -p /home/n/n_herr03/musket-build/build/benchmark && \

cd /home/n/n_herr03/musket-build/src/ && \
nvcc aco_iroulette_0.cu -o aco_iroulette_0 -gencode arch=compute_35,code=sm_35 && \

srun /home/n/n_herr03/musket-build/build/benchmark/bin/aco_iroulette_0 1 15 1 1024
