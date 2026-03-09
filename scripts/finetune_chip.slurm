#!/bin/bash
#SBATCH --job-name=chip_ft
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --array=0-1
#SBATCH --output=logs/chip_ft_%A_%a.out
#SBATCH --error=logs/chip_ft_%A_%a.err

set -euo pipefail

# Parse --name from args; everything else forwarded to finetune_chip.py
PASSTHROUGH_ARGS=("$@")
MODEL_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --name) MODEL_NAME="$2"; shift 2;;
        *) shift;;
    esac
done

# Map array index to target
TARGETS=(gfp polii)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}

# Rename SLURM job if --name provided
if [ -n "$MODEL_NAME" ]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="${MODEL_NAME}_${TARGET}" 2>/dev/null
fi

echo "=== Job $SLURM_JOB_ID | Array task $SLURM_ARRAY_TASK_ID | Target: $TARGET ==="
[ -n "$MODEL_NAME" ] && echo "Name: $MODEL_NAME"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
date

PROJECT_DIR=/grid/wsbs/home_norepl/pmantill/Worm_ChiP
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

module load cuda12.3/toolkit/12.3.2
module load alphagenome/v1

# NVIDIA libs for JAX GPU
NVIDIA_LIBS="$PROJECT_DIR/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cuda_cupti/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/nvjitlink/lib:$NVIDIA_LIBS/nccl/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

python scripts/finetune_chip.py \
    --config configs/chip_joint.json \
    --target "$TARGET" \
    "${PASSTHROUGH_ARGS[@]}"

echo "=== Done ==="
date
