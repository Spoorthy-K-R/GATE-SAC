#!/bin/bash
#SBATCH --job-name=gatesac
#SBATCH --account=132702223356        # Grace: 132702223356 | FASTER: 142702220759
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-44
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%A_%a.out

module restore gatesac_env
source $SCRATCH/venvs/ssl/bin/activate
export MPLCONFIGDIR=$SCRATCH/tmp/mpl
cd "$SLURM_SUBMIT_DIR"

CONFIGS=(b1_sac b2_saclag b3_mbpo_h1 b3_mbpo_h3 b3_mbpo_h5 b3_mbpo_h7 a1_penalty a2_gate g_gatesac)
N_SEEDS=5
CFG=${CONFIGS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))

python -m gate_sac.train \
    --config configs/$CFG.yaml \
    --splits splits/waymo.json \
    --demos runs/demos.npz \
    --bc runs/bc/actor.pt \
    --seed $SEED \
    --out runs/sweep/${CFG}_s${SEED}
