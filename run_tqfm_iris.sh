#!/bin/bash
#SBATCH --job-name=tqfm_iris
#SBATCH --output=/data/%u/logs/tqfm_iris_%j.out
#SBATCH --error=/data/%u/logs/tqfm_iris_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --array=1-5

cd /home/$USER/SVQSVM/

# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000

# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000

# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_iris.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
