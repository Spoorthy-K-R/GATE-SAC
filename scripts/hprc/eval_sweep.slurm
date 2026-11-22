#!/bin/bash
#SBATCH --job-name=gatesac_eval
#SBATCH --account=132702223356
#SBATCH --array=0-44
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%A_%a.out

module restore gatesac_env
source $SCRATCH/venvs/ssl/bin/activate
export MPLCONFIGDIR=$SCRATCH/tmp/mpl
cd "$SLURM_SUBMIT_DIR"

CONFIGS=(b1_sac b2_saclag b3_mbpo_h1 b3_mbpo_h3 b3_mbpo_h5 b3_mbpo_h7 a1_penalty a2_gate g_gatesac)
N_SEEDS=5
CFG=${CONFIGS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
RUN_DIR="runs/sweep/${CFG}_s${SEED}"

# Fallback: if a task hit the 6h limit before writing model_final.zip, copy its latest checkpoint
if [ ! -f "${RUN_DIR}/model_final.zip" ]; then
    LATEST=$(ls -1v "${RUN_DIR}"/model_*_steps.zip 2>/dev/null | tail -n 1)
    cp "$LATEST" "${RUN_DIR}/model_final.zip"
fi

# 1. Evaluate on held-out TEST split (82 scenarios)
python -m gate_sac.eval.evaluate \
    --splits splits/waymo.json \
    --split test \
    --model "${RUN_DIR}/model_final.zip" \
    --out "runs/eval/${CFG}_s${SEED}_test.json"

# 2. Evaluate on mined BC-FAILURE split (139 scenarios)
python -m gate_sac.eval.evaluate \
    --splits splits/waymo.json \
    --split failure \
    --model "${RUN_DIR}/model_final.zip" \
    --out "runs/eval/${CFG}_s${SEED}_failure.json"
