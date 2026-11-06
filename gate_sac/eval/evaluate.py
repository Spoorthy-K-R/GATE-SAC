"""Closed-loop evaluation on a scenario split, with per-km safety metrics.

    python -m gate_sac.eval.evaluate --splits splits/waymo_sample.json --split test --bc runs/bc/actor.pt
    python -m gate_sac.eval.evaluate --splits splits/waymo_sample.json --split test --model runs/G_s0/model_100000.zip
"""
import argparse
import json

import numpy as np

from gate_sac.data.splits import load_splits
from gate_sac.envs.scenario_env import GateEnv


def run_episodes(env, act_fn, scenarios, episodes_per_scenario=1):
    env.set_scenarios(scenarios, sequential=True)
    episodes = []
    for _ in range(len(scenarios) * episodes_per_scenario):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(act_fn(obs))
            done = terminated or truncated
        episodes.append(info["ep_metrics"])
    return episodes


def summarize(episodes):
    km = sum(e["km"] for e in episodes)
    mean = lambda k: float(np.mean([e[k] for e in episodes]))
    return dict(
        episodes=len(episodes),
        success_rate=mean("success"),
        route_completion=mean("route_completion"),
        collisions_per_km=sum(e["crash"] for e in episodes) / km,
        out_of_road_per_km=sum(e["out_of_road"] for e in episodes) / km,
        cost_per_km=sum(e["cost"] for e in episodes) / km,
        avg_speed_kmh=mean("avg_speed_kmh"),
        ret=mean("ret"),
        km=km,
    )


def policy_act_fn(path, kind):
    import torch
    if kind == "bc":
        from gate_sac.bc.train_bc import load_bc_actor
        actor = load_bc_actor(path)
        return lambda obs: actor(torch.as_tensor(obs[None], dtype=torch.float32), deterministic=True)[0].detach().numpy()
    from gate_sac.agents.sac_lag import SACLag
    model = SACLag.load(path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--split", default="test")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bc")
    g.add_argument("--model")
    g.add_argument("--expert", action="store_true")
    p.add_argument("--episodes-per-scenario", type=int, default=1)
    p.add_argument("--out")
    a = p.parse_args()

    splits = load_splits(a.splits)
    env = GateEnv(splits["data_dir"])
    if a.expert:
        from gate_sac.envs.expert import LogTrackingExpert
        act_fn = LogTrackingExpert(env).act
    else:
        act_fn = policy_act_fn(a.bc or a.model, "bc" if a.bc else "sac")
    episodes = run_episodes(env, act_fn, splits[a.split], a.episodes_per_scenario)
    env.close()
    result = dict(summary=summarize(episodes), episodes=episodes)
    print(json.dumps(result["summary"], indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
