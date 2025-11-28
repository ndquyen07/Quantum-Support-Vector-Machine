#!/bin/bash
#SBATCH --job-name=tqfm_optimizer
#SBATCH --output=/data/%u/logs/tqfm_%j.out
#SBATCH --error=/data/%u/logs/tqfm_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --array=1-10


export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"

cd /home/$USER/SVQSVM/

# python3.13 main_tqfm_digits.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz RealAmplitudes --optimizer COBYLA --maxiter 100000
# python3.13 main_tqfm_digits.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz RealAmplitudes --optimizer SPSA --maxiter 100000
# python3.13 main_tqfm_digits.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --type_ansatz RealAmplitudes --optimizer ADAM --maxiter 100000

