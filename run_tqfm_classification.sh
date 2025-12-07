#!/bin/bash
#SBATCH --job-name=tqfm_classification
#SBATCH --output=/home/%u/logs/tqfm_%j.out
#SBATCH --error=/home/%u/logs/tqfm_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048
#SBATCH --array=1-5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate svqsvm

python3.13 main_tqfm_classification.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
    

# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz TwoLocal --optimizer COBYLA --maxiter 100000

# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000

# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_cancer.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --type_ansatz EfficientSU2 --optimizer COBYLA --maxiter 100000
