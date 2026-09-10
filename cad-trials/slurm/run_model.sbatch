#!/bin/bash
# Run a single model wrapper task from the manifest
# Usage: sbatch cad-trials/slurm/run_model.sbatch [manifest_path]
# Reads line $SLURM_ARRAY_TASK_ID from manifest and runs the model wrapper.
# Resumability is handled inside the wrapper (already_done), so sbatch doesn't need dedup.

#SBATCH --partition main
#SBATCH --qos normal
#SBATCH --gpus rtx_3090:1
#SBATCH --cpus-per-task 4
#SBATCH --mem 24G
#SBATCH --time 8:00:00
#SBATCH --array=1-9
#SBATCH -o cad-trials/results/slurm-%A_%a.out

set -e

# Manifest path (default: cad-trials/results/manifest.tsv)
MANIFEST="${1:-cad-trials/results/manifest.tsv}"

# Go to repo root
cd "$(git rev-parse --show-toplevel)"

# Read the task line from manifest
# Line numbers are 1-indexed, but arrays are typically 1-indexed in slurm
LINE_NUM=$SLURM_ARRAY_TASK_ID
line=$(sed -n "${LINE_NUM}p" "$MANIFEST")

if [ -z "$line" ]; then
    echo "ERROR: Line $LINE_NUM not found in $MANIFEST"
    exit 1
fi

# Parse tab-separated columns: model, input_path, input_kind, n_samples, env
IFS=$'\t' read -r model input_path input_kind n_samples env <<< "$line"

echo "Task $SLURM_ARRAY_TASK_ID: model=$model, input_kind=$input_kind, env=$env"

# Activate conda environment
conda activate "$env"

# Run the model wrapper
python -m cad_trials.models."$model" \
    --inputs "$input_path" \
    --out-dir cad-trials/results/"$model" \
    --gt cad-trials/results/prepared/part/normalized.stl \
    --n-samples "$n_samples" \
    --runs-path cad-trials/results/runs.jsonl \
    --problem p1
