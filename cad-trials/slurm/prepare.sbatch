#!/bin/bash
# Prepare problem 1 data: stl → renders + point clouds + normalized mesh
# Usage: sbatch cad-trials/slurm/prepare.sbatch

#SBATCH --partition main
#SBATCH --qos normal
#SBATCH --cpus-per-task 4
#SBATCH --mem 24G
#SBATCH --time 1:00:00
#SBATCH -o cad-trials/results/slurm-prepare.out

set -e

# Go to repo root
cd "$(git rev-parse --show-toplevel)"

# Set up Python path and conda
export PYTHONPATH="$PWD/cad-trials:$PYTHONPATH"
module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate probe-common

# Run prepare_problem1
python -m cad_trials.data.prepare_problem1 \
    --stl output/fine.STL \
    --out-dir cad-trials/results/prepared/part
