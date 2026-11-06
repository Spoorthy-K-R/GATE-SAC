"""Train one method of the experiment matrix (B1, B2, B3-H*, A1, A2, G) for one seed.

    python -m gate_sac.train --config configs/g_gatesac.yaml --splits splits/waymo.json \
        --demos runs/demos.npz --bc runs/bc/actor.pt --seed 0 --out runs/G_s0

Checkpoints are written every `checkpoint_every` env steps; evaluate them with gate_sac.eval.evaluate
(MetaDrive allows one simulator per process, so evaluation runs in separate processes).
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from gate_sac.agents.sac_lag import SACLag
from gate_sac.bc.train_bc import ACTOR_NET_ARCH
from gate_sac.data.splits import load_splits
from gate_sac.envs.scenario_env import GateEnv


def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        cfg = deep_update(load_config(Path(path).parent / base), cfg)
    return cfg


def training_scenarios(splits, failure_frac):
    """Failure-set scenarios drawn with total probability `failure_frac`, the rest of train otherwise."""
    failure = [s for s in splits["failure"] if s in set(splits["train"])]
    normal = [s for s in splits["train"] if s not in set(failure)]
    if not failure or not normal:
        return splits["train"], None
    probs = [failure_frac / len(failure)] * len(failure) + [(1 - failure_frac) / len(normal)] * len(normal)
    return failure + normal, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--demos")
    p.add_argument("--bc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--total-steps", type=int)
    a = p.parse_args()

    cfg = load_config(a.config)
    tr = cfg["train"]
    total_steps = a.total_steps or tr["total_steps"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(dict(cfg=cfg, args=vars(a)), f, indent=1)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits = load_splits(a.splits)
    env = GateEnv(splits["data_dir"], seed=a.seed, env_config=cfg.get("env"))
    env.set_scenarios(*training_scenarios(splits, tr["failure_frac"]))

    agent_cfg = dict(cfg["agent"])
    if tr.get("bc_init", True):
        assert a.bc, "this config starts from BC: pass --bc"
        agent_cfg["bc_actor_path"] = a.bc
    else:
        agent_cfg.update(warmup_updates=0, bc_reg=0.0)
    model = SACLag(
        "MlpPolicy", env, seed=a.seed, verbose=0, device="cuda" if torch.cuda.is_available() else "cpu",
        learning_starts=tr.get("learning_starts", 0),
        policy_kwargs=dict(net_arch=dict(pi=ACTOR_NET_ARCH, qf=[256, 256])),
        **cfg["sac"], **agent_cfg,
    )
    if a.demos and tr.get("seed_buffer_with_demos", True):
        model.load_demos(dict(np.load(a.demos)))
    model.set_logger(configure(str(out), ["csv", "tensorboard", "stdout"] if tr.get("stdout") else ["csv", "tensorboard"]))
    ckpt = CheckpointCallback(save_freq=tr["checkpoint_every"], save_path=str(out), name_prefix="model")
    model.learn(total_timesteps=total_steps, callback=ckpt, log_interval=tr.get("log_interval", 10))
    model.save(out / "model_final")
    env.close()


if __name__ == "__main__":
    main()
