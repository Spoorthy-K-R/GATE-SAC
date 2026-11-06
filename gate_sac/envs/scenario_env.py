"""MetaDrive ScenarioEnv (ScenarioNet / Waymo scenarios) with the GATE-SAC reward and cost.

Reward  r = progress + speed - lane_deviation (+ success bonus).  No crash penalty: crashes and
        leaving the road enter only through the cost signal c, handled by the Lagrangian cost critic.
Cost    c = 1 on vehicle/object/human crash or out-of-road (MetaDrive's built-in cost), else 0.

`GateEnv` wraps the simulator so that every reset draws a scenario from a given index list
(train / failure / test splits) and every finished episode reports per-km safety metrics.
"""
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv

GATE_ENV_CONFIG = dict(
    progress_reward=1.0,  # per metre of progress along the logged route
    speed_reward=0.1,  # * min(speed / target_speed, 1)
    target_speed_kmh=40.0,
    lane_dev_penalty=0.5,  # * |lateral offset| / max_lateral_dist
    success_reward=5.0,
    terminal_penalty=0.0,  # reward on crash / out-of-road; 0 so the cost is not double counted
)


class GateScenarioEnv(ScenarioEnv):

    @classmethod
    def default_config(cls):
        config = super(GateScenarioEnv, cls).default_config()
        config.update(GATE_ENV_CONFIG, allow_add_new_key=True)
        config.update(
            dict(
                crash_vehicle_done=True,
                crash_object_done=True,
                crash_human_done=True,
                horizon=200,
                log_level=50,
            )
        )
        return config

    def reward_function(self, vehicle_id: str):
        cfg = self.config
        vehicle = self.agents[vehicle_id]
        nav = vehicle.navigation

        progress = nav.current_longitude - nav.last_longitude
        speed = min(vehicle.speed_km_h / cfg["target_speed_kmh"], 1.0)
        lane_dev = min(abs(nav.current_lateral) / cfg["max_lateral_dist"], 1.0)
        reward = cfg["progress_reward"] * progress + cfg["speed_reward"] * speed - cfg["lane_dev_penalty"] * lane_dev

        crashed = vehicle.crash_vehicle or vehicle.crash_object or vehicle.crash_human
        if self._is_arrive_destination(vehicle):
            reward = cfg["success_reward"]
        elif crashed or self._is_out_of_road(vehicle):
            reward = -cfg["terminal_penalty"]

        step_info = dict(
            route_completion=nav.route_completion,
            lateral_dist=nav.current_lateral,
            r_progress=progress,
            r_lane_dev=lane_dev,
            scenario_index=self.engine.current_seed,
        )
        return float(reward), step_info


def dataset_size(data_dir) -> int:
    with open(Path(data_dir) / "dataset_summary.pkl", "rb") as f:
        return len(pickle.load(f))


def resolve_data_dir(data_dir):
    """`waymo` / `nuscenes` select the small sample sets bundled with MetaDrive."""
    if data_dir in ("waymo", "nuscenes"):
        return AssetLoader.file_path(data_dir, unix_style=False)
    return str(Path(data_dir).expanduser())


class GateEnv(gym.Wrapper):
    """Scenario sampling over an index list + per-episode safety metrics in `info["ep_metrics"]`.

    MetaDrive allows one simulator per process, so train and eval splits are switched with
    `set_scenarios` instead of creating a second env.
    """

    def __init__(self, data_dir, scenarios=None, seed=0, sequential=False, env_config=None):
        data_dir = resolve_data_dir(data_dir)
        n_total = dataset_size(data_dir)
        config = dict(data_directory=data_dir, start_scenario_index=0, num_scenarios=n_total)
        config.update(env_config or {})
        super().__init__(GateScenarioEnv(config))
        self.rng = np.random.default_rng(seed)
        self.set_scenarios(range(n_total) if scenarios is None else scenarios, sequential)

    def set_scenarios(self, scenarios, sequential=False, probs=None):
        self.scenarios = np.asarray(list(scenarios), dtype=int)
        self.sequential = sequential
        self.probs = probs
        self._next = 0

    @property
    def agent(self):
        return self.env.agent

    def reset(self, *, seed=None, options=None):
        if seed is None:
            if self.sequential:
                seed = int(self.scenarios[self._next % len(self.scenarios)])
                self._next += 1
            else:
                seed = int(self.rng.choice(self.scenarios, p=self.probs))
        obs, info = self.env.reset(seed=seed)
        self._pos = np.array(self.agent.position)
        self._ep = dict(scenario=seed, ret=0.0, cost=0.0, metres=0.0, steps=0, speed_sum=0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = np.array(self.agent.position)
        ep = self._ep
        ep["metres"] += float(np.linalg.norm(pos - self._pos))
        self._pos = pos
        ep["ret"] += reward
        ep["cost"] += info.get("cost", 0.0)
        ep["steps"] += 1
        ep["speed_sum"] += self.agent.speed_km_h
        if terminated or truncated:
            km = max(ep["metres"] / 1000.0, 1e-6)
            info["ep_metrics"] = dict(
                scenario=ep["scenario"],
                ret=ep["ret"],
                cost=ep["cost"],
                km=km,
                steps=ep["steps"],
                avg_speed_kmh=ep["speed_sum"] / ep["steps"],
                success=float(info.get("arrive_dest", False)),
                crash=float(info.get("crash", False)),
                out_of_road=float(info.get("out_of_road", False)),
                route_completion=float(info.get("route_completion", 0.0)),
                timeout=float(truncated and not terminated),
            )
        return obs, reward, terminated, truncated, info
