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

# Activate conda environment
conda activate probe-common

# Run prepare_problem1
cd "$(git rev-parse --show-toplevel)"
python -m cad_trials.data.prepare_problem1 \
    --stl output/fine.STL \
    --out-dir cad-trials/results/prepared/part
