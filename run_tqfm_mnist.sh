#!/bin/bash
#SBATCH --job-name=tqfm_mnist
#SBATCH --output=/data/%u/logs/tqfm_mnist_%j.out
#SBATCH --error=/data/%u/logs/tqfm_mnist_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --array=1-10

cd /home/$USER/SVQSVM/

# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000

# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000

# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_mnist.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
