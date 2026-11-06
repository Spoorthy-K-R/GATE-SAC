"""Collect demonstrations from the log-tracking expert on the train split.

Executed actions get Gaussian noise (DART-style) so the data covers small recoveries; the BC label
is always the clean expert action at the visited state. Every transition is saved, so the same file
also seeds the real replay buffer of the RL stage.

    python -m gate_sac.bc.collect --splits splits/waymo_sample.json --episodes-per-scenario 20 --out runs/demos.npz
"""
import argparse
from pathlib import Path

import numpy as np

from gate_sac.data.splits import load_splits
from gate_sac.envs.expert import LogTrackingExpert
from gate_sac.envs.scenario_env import GateEnv


def collect(env, expert, scenarios, episodes_per_scenario, noise_std, seed=0):
    rng = np.random.default_rng(seed)
    env.set_scenarios(scenarios, sequential=True)
    keys = ["obs", "expert_action", "action", "reward", "cost", "next_obs", "terminated", "truncated"]
    data = {k: [] for k in keys}
    episodes = []
    for ep in range(len(scenarios) * episodes_per_scenario):
        obs, _ = env.reset()
        # First pass over each scenario is noise-free so the clean expert is in the data too.
        std = 0.0 if ep < len(scenarios) else noise_std
        done = False
        while not done:
            a_exp = expert.act(obs)
            a = np.clip(a_exp + std * rng.standard_normal(2), -1, 1).astype(np.float32)
            next_obs, r, terminated, truncated, info = env.step(a)
            for k, v in zip(keys, [obs, a_exp, a, r, info.get("cost", 0.0), next_obs, terminated, truncated]):
                data[k].append(v)
            obs, done = next_obs, terminated or truncated
        episodes.append(info["ep_metrics"])
    return {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}, episodes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--episodes-per-scenario", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from gate_sac.eval.evaluate import summarize

    splits = load_splits(a.splits)
    env = GateEnv(splits["data_dir"])
    data, episodes = collect(env, LogTrackingExpert(env), splits[a.split], a.episodes_per_scenario, a.noise_std)
    env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data)
    print(f"{len(data['obs'])} transitions -> {a.out}")
    print("expert (incl. noisy episodes):", summarize(episodes))
