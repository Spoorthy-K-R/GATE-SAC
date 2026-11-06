"""Log-tracking expert: drives the ego car along the recorded human (Waymo) trajectory.

Waymo logs contain poses, not steering/throttle. Instead of fitting an inverse-dynamics model we
close the loop in the simulator: pure-pursuit steering toward the logged position a short time
ahead, and a P-controller on the logged speed. The (obs, action) pairs it produces are exactly
consistent with MetaDrive's physics, and the path/speed profile is the human's.

MetaDrive's steering convention is +1 = left (counter-clockwise), -1 = right.
"""
import numpy as np


class LogTrackingExpert:

    def __init__(self, env, lookahead_steps=5, min_lookahead_m=4.0, wheelbase=2.8, speed_gain=0.25):
        self.env = env  # GateEnv or GateScenarioEnv
        self.lookahead_steps = lookahead_steps
        self.min_lookahead_m = min_lookahead_m
        self.wheelbase = wheelbase
        self.speed_gain = speed_gain

    def _log(self):
        engine = self.env.unwrapped.engine
        scenario = engine.data_manager.current_scenario
        state = scenario["tracks"][scenario["metadata"]["sdc_id"]]["state"]
        valid = state["valid"].astype(bool)
        return state["position"][valid, :2], np.linalg.norm(state["velocity"][valid], axis=-1), engine.episode_step

    def act(self, obs=None):
        positions, speeds, t = self._log()
        vehicle = self.env.unwrapped.agent
        pos = np.asarray(vehicle.position)
        heading = vehicle.heading_theta

        # Look ahead from the logged point nearest to the car (not from time t), so a car that has
        # fallen behind the log steers back onto the path instead of cutting corners.
        nearest = int(np.argmin(np.linalg.norm(positions - pos, axis=-1)))
        target = min(nearest + self.lookahead_steps, len(positions) - 1)
        while target < len(positions) - 1 and np.linalg.norm(positions[target] - pos) < self.min_lookahead_m:
            target += 1
        delta = positions[target] - pos
        ld = max(np.linalg.norm(delta), 1e-3)
        alpha = np.arctan2(delta[1], delta[0]) - heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        steer_angle = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), ld)
        max_steer = np.deg2rad(vehicle.config["max_steering"])
        steer = np.clip(steer_angle / max_steer, -1.0, 1.0)

        target_speed = speeds[min(t + 1, len(speeds) - 1)]
        throttle = np.clip(self.speed_gain * (target_speed - vehicle.speed), -1.0, 1.0)
        return np.array([steer, throttle], dtype=np.float32)
