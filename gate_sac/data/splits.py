"""Make train / test scenario splits for a ScenarioNet database.

Scenarios where the logged ego car barely moves are dropped: there is nothing to drive.

    python -m gate_sac.data.splits --data waymo --out splits/waymo_sample.json
    python -m gate_sac.data.splits --data ~/data/waymo_scenarionet --out splits/waymo.json --test-frac 0.2
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scenarionet.common_utils import read_dataset_summary

from gate_sac.envs.scenario_env import resolve_data_dir


def make_splits(data_dir, test_frac=0.2, min_ego_distance=20.0, seed=0):
    summary, lookup, _ = read_dataset_summary(resolve_data_dir(data_dir))
    keep = []
    for idx, name in enumerate(lookup):
        meta = summary[name]
        if meta["object_summary"][meta["sdc_id"]]["moving_distance"] >= min_ego_distance:
            keep.append(idx)
    rng = np.random.default_rng(seed)
    keep = rng.permutation(keep).tolist()
    n_test = max(1, int(round(test_frac * len(keep)))) if len(keep) > 1 else 0
    return dict(
        data_dir=str(data_dir),
        n_total=len(lookup),
        dropped=sorted(set(range(len(lookup))) - set(keep)),
        train=sorted(keep[n_test:]),
        test=sorted(keep[:n_test]),
        failure=[],  # filled in by gate_sac.bc.mine_failures
    )


def load_splits(path):
    with open(path) as f:
        return json.load(f)


def save_splits(splits, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--min-ego-distance", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    s = make_splits(a.data, a.test_frac, a.min_ego_distance, a.seed)
    save_splits(s, a.out)
    print(f"{a.out}: {len(s['train'])} train, {len(s['test'])} test, {len(s['dropped'])} dropped")
