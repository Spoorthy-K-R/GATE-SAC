"""Find the train scenarios where the BC policy fails; they become the RL failure set.

Failure = crash, out-of-road, or route completion below a threshold, in any of the rollouts.

    python -m gate_sac.bc.mine_failures --splits splits/waymo_sample.json --bc runs/bc/actor.pt
"""
import argparse
from collections import defaultdict

from gate_sac.data.splits import load_splits, save_splits
from gate_sac.envs.scenario_env import GateEnv
from gate_sac.eval.evaluate import policy_act_fn, run_episodes, summarize


def is_failure(ep, min_route_completion):
    return ep["crash"] or ep["out_of_road"] or ep["route_completion"] < min_route_completion


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--bc", required=True)
    p.add_argument("--min-route-completion", type=float, default=0.8)
    a = p.parse_args()

    splits = load_splits(a.splits)
    env = GateEnv(splits["data_dir"])
    episodes = run_episodes(env, policy_act_fn(a.bc, "bc"), splits["train"])
    env.close()

    failed = defaultdict(bool)
    for ep in episodes:
        failed[ep["scenario"]] |= bool(is_failure(ep, a.min_route_completion))
    splits["failure"] = sorted(s for s, f in failed.items() if f)
    save_splits(splits, a.splits)
    print("BC on train split:", summarize(episodes))
    print(f"failure set: {len(splits['failure'])}/{len(splits['train'])} train scenarios -> {a.splits}")
