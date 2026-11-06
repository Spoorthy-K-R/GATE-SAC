#!/bin/bash
# Full experiment matrix as a job array: 9 configs x 5 seeds = 45 tasks.
#   sbatch scripts/hprc/sweep.slurm
# Edit SPLITS / DEMOS / BC / ENV paths for your HPRC scratch space first.
#SBATCH --job-name=gatesac
#SBATCH --array=0-44
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out

CONFIGS=(b1_sac b2_saclag b3_mbpo_h1 b3_mbpo_h3 b3_mbpo_h5 b3_mbpo_h7 a1_penalty a2_gate g_gatesac)
N_SEEDS=5
CFG=${CONFIGS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))

SPLITS=splits/waymo.json
DEMOS=runs/demos.npz
BC=runs/bc/actor.pt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ssl
cd "$SLURM_SUBMIT_DIR"
python -m gate_sac.train --config configs/$CFG.yaml --splits $SPLITS --demos $DEMOS --bc $BC \
    --seed $SEED --out runs/sweep/${CFG}_s${SEED}
