#!/bin/bash
#SBATCH --job-name aco_iroulette-GPU-nvprof-nodes-1-gpu-1
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --exclusive
#SBATCH --output /home/fwrede/musket-build/aco_iroulette/CUDA/out/aco_iroulette-nvprof-nodes-1-gpu-1.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 01:00:00
#SBATCH -A p_algcpugpu
#SBATCH --gres gpu:4

export OMP_NUM_THREADS=24

RUNS=1
for ((i=1;i<=RUNS;i++)); do
	srun nvprof /home/fwrede/out/mnp/aco_iroulette-cuda-%p.out /home/fwrede/build/mnp/aco_iroulette/cuda/bin/aco_iroulette_0
done
