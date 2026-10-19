#!/bin/bash
# Supply --account and --partition on the sbatch command for your cluster.
#SBATCH --job-name=active-sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
# Activate the environment before sbatch, or pass PYTHON=/path/to/python.
exec "${PYTHON:-python}" -u -m active_sample run "$@"
