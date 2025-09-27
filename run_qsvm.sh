#!/bin/bash
#SBATCH --job-name=qsvm
#SBATCH --output=/home/%u/logs/qsvm_%j.out
#SBATCH --error=/home/%u/logs/qsvm_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --array=1-100
#SBATCH --time=00:30:00

cd /home/$USER/SVQSVM/

python3.13 main.py --run_id ${SLURM_ARRAY_TASK_ID} --type_ansatz="custom1" --depth=1

