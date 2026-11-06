# GATE-SAC

Uncertainty-gated, BC-initialized, Lagrangian SAC fine-tuning on ScenarioNet (Waymo) scenarios in MetaDrive.
See `GATE-SAC-project-proposal.pdf` for the project description.

## Environment

The conda env `ssl` (Python 3.10) has everything installed: `metadrive-simulator` 0.4.3, `stable-baselines3` 2.9,
`scenarionet`, `torch` 2.8 (CUDA). MetaDrive's assets, including small Waymo (3) and nuScenes (10) sample
sets, were pulled with `python -m metadrive.pull_asset`.

```bash
conda activate ssl
```

## Layout

| Path | What |
|---|---|
| `gate_sac/envs/scenario_env.py` | `GateScenarioEnv`: reward = progress + speed - lane deviation (no crash penalty); cost = crash / out-of-road; crash ends the episode. `GateEnv`: scenario sampling per split + per-km metrics |
| `gate_sac/envs/expert.py` | Log-tracking expert: pure pursuit on the logged human path + P-control on logged speed. Gives BC action labels Waymo logs lack |
| `gate_sac/data/splits.py` | train / test split (drops scenarios where the logged ego car barely moves) |
| `gate_sac/bc/` | `collect.py` demos (DART-style noise), `train_bc.py` BC = an SB3 SAC actor, `mine_failures.py` failure set |
| `gate_sac/models/ensemble.py` | 4-member probabilistic MLP ensemble: Δs, r (Gaussian), cost, done; disagreement u |
| `gate_sac/agents/rollout.py` | Imagined rollouts: fixed-H / running-percentile gate / MOPO penalty, n-step targets |
| `gate_sac/agents/sac_lag.py` | `SACLag` (subclass of SB3 SAC): cost critic, λ_c dual ascent on real cost/km, BC warm-up + annealed BC regularizer, real/imagined batch mixing |
| `gate_sac/train.py` | Train one config × seed |
| `gate_sac/eval/evaluate.py` | Closed-loop eval: success, route completion, collisions/km, out-of-road/km, speed, return |
| `configs/` | Experiment matrix: `b1_sac`, `b2_saclag`, `b3_mbpo_h{1,3,5,7}`, `a1_penalty`, `a2_gate`, `g_gatesac` |
| `scripts/hprc/sweep.slurm` | Job array: 9 configs × 5 seeds |

## Pipeline

```bash
DATA=nuscenes            # bundled sample for development; a ScenarioNet Waymo database dir later
S=splits/dev.json
python -m gate_sac.data.splits --data $DATA --out $S --test-frac 0.25
python -m gate_sac.bc.collect --splits $S --episodes-per-scenario 10 --out runs/demos.npz
python -m gate_sac.bc.train_bc --demos runs/demos.npz --out runs/bc/actor.pt
python -m gate_sac.eval.evaluate --splits $S --split test --bc runs/bc/actor.pt          # B0
python -m gate_sac.bc.mine_failures --splits $S --bc runs/bc/actor.pt                   # fills splits["failure"]
python -m gate_sac.train --config configs/g_gatesac.yaml --splits $S --demos runs/demos.npz \
    --bc runs/bc/actor.pt --seed 0 --out runs/G_s0
python -m gate_sac.eval.evaluate --splits $S --split test --model runs/G_s0/model_final.zip
```

Training logs (`progress.csv`, TensorBoard) include `episode/*` safety metrics, `train/lambda_c`, and
`model/{truncation_rate, mean_length, u_norm_mean, threshold, holdout_mse}` for the gate analysis.

## Real Waymo data

1. Register for the Waymo Open Dataset and download the Motion dataset `scenario/*.tfrecord` files.
2. Converting requires TensorFlow and `waymo-open-dataset` (not installed in `ssl`, to keep it lean):
   `python -m scenarionet.convert_waymo -d ~/data/waymo_sn --raw_data_path <tfrecords> --num_workers 8`
3. `python -m gate_sac.data.splits --data ~/data/waymo_sn --out splits/waymo.json`

## Conventions

- MetaDrive steering is **+1 = left**, −1 = right (the proposal states the opposite; only the sign convention differs).
- One MetaDrive simulator per process: evaluate checkpoints in separate processes.
