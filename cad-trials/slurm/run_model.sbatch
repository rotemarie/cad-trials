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
# Array size must match the manifest line count. Submit with an explicit override:
#   sbatch --array=1-$(wc -l < cad-trials/results/manifest.tsv) cad-trials/slurm/run_model.sbatch
# The directive below is a default for the current 9-task manifest.
#SBATCH --array=1-9
#SBATCH -o cad-trials/results/slurm-%A_%a.out

set -e

# Manifest path (default: cad-trials/results/manifest.tsv)
MANIFEST="${1:-cad-trials/results/manifest.tsv}"

# Go to repo root
cd "$(git rev-parse --show-toplevel)"

# Set up Python path and conda
export PYTHONPATH="$PWD/cad-trials:$PYTHONPATH"
module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"

# Read the task line from manifest
# Line numbers are 1-indexed, but arrays are typically 1-indexed in slurm
LINE_NUM=$SLURM_ARRAY_TASK_ID
line=$(sed -n "${LINE_NUM}p" "$MANIFEST")

if [ -z "$line" ]; then
    echo "ERROR: Line $LINE_NUM not found in $MANIFEST"
    exit 1
fi

# Parse tab-separated columns: part, model, input_path, input_kind, n_samples, env
IFS=$'\t' read -r part model input_path input_kind n_samples env <<< "$line"

echo "Task $SLURM_ARRAY_TASK_ID: part=$part, model=$model, input_kind=$input_kind, env=$env"

# Ground truth lives next to the input, inside that part's prepared dir — deriving
# it keeps multi-part manifests from scoring every part against part 1's mesh.
gt="$(dirname "$input_path")/normalized.stl"

# Each array task writes its OWN ledger shard. Concurrent unlocked appends to a
# single runs.jsonl on NFS can interleave and tear a line. One array task owns one
# (part, model, input) pair and all its samples, so its shard is all it needs to
# read for the resume check. Reports read the whole directory:
#   build_report('cad-trials/results/runs.d', ...)
runs_path="cad-trials/results/runs.d/${SLURM_ARRAY_TASK_ID}.jsonl"
mkdir -p "$(dirname "$runs_path")"

# Activate conda environment
conda activate "$env"

# Run the model wrapper
python -m cad_trials.models."$model" \
    --inputs "$input_path" \
    --out-dir cad-trials/results/"$model" \
    --gt "$gt" \
    --part "$part" \
    --n-samples "$n_samples" \
    --runs-path "$runs_path" \
    --problem p1
