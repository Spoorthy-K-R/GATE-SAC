"""Render side-by-side top-down driving videos (Expert vs. Policy) with telemetry HUD.

Usage:
    .venv/bin/python -m gate_sac.eval.record_video --data waymo --scenarios 0 1 2 --bc runs/bc/actor.pt --out-dir runs/videos
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from gate_sac.envs.expert import LogTrackingExpert
from gate_sac.envs.scenario_env import GateEnv


def load_standalone_bc(path, device="cpu"):
    """Load BC actor weights directly into a pure PyTorch MLP (no SB3 import required)."""
    ckpt = torch.load(path, map_location=device)
    obs_dim = ckpt["obs_dim"]
    arch = ckpt.get("net_arch", {}).get("pi", [256, 256, 256])
    layers = []
    in_dim = obs_dim
    for h in arch:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    latent_pi = nn.Sequential(*layers).to(device)
    mu = nn.Linear(in_dim, 2).to(device)
    sd = ckpt["actor"]
    latent_pi.load_state_dict({k.replace("latent_pi.", ""): v for k, v in sd.items() if k.startswith("latent_pi.")})
    mu.load_state_dict({k.replace("mu.", ""): v for k, v in sd.items() if k.startswith("mu.")})
    latent_pi.eval()
    mu.eval()

    def act(obs):
        with torch.no_grad():
            x = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
            return torch.tanh(mu(latent_pi(x)))[0].cpu().numpy()

    return act


def annotate_frame(frame, title, step, speed_kmh, action, rc, status):
    """Draw a clean telemetry HUD banner at the top and bottom of a top-down frame."""
    img = frame.copy()
    h, w, _ = img.shape

    # Top banner background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 76), (20, 24, 32), -1)
    cv2.rectangle(overlay, (0, h - 38), (w, h), (20, 24, 32), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    # Status color (RGB)
    if "SUCCESS" in status:
        badge_color = (46, 204, 113)
    elif "CRASH" in status or "OUT OF ROAD" in status:
        badge_color = (231, 76, 60)
    elif "TIMEOUT" in status:
        badge_color = (243, 156, 18)
    else:
        badge_color = (52, 152, 219)

    cv2.putText(img, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (w - 195, 8), (w - 10, 34), badge_color, -1)
    cv2.putText(img, status[:16], (w - 187, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    telem = f"Step: {step:3d}  |  Speed: {speed_kmh:4.1f} km/h  |  Route: {rc * 100:4.1f}%"
    cv2.putText(img, telem, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 220, 235), 1, cv2.LINE_AA)

    steer, thr = float(action[0]), float(action[1])
    ctrl = f"Steer: {steer:+.2f}   Throttle/Brake: {thr:+.2f}"
    cv2.putText(img, ctrl, (14, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 220, 235), 1, cv2.LINE_AA)
    return img


def rollout_frames(env, act_fn, scenario_idx, title, screen_size=(520, 520)):
    obs, _ = env.reset(seed=scenario_idx)
    frames = []
    done = False
    step = 0
    rc = 0.0
    status = "DRIVING"
    last_action = np.zeros(2, dtype=np.float32)

    while not done:
        action = act_fn(obs)
        last_action = action
        obs, _, terminated, truncated, info = env.step(action)
        step += 1
        rc = float(info.get("route_completion", rc))
        speed = float(env.agent.speed_km_h)
        done = terminated or truncated
        if done:
            if info.get("arrive_dest", False):
                status = "SUCCESS"
            elif info.get("crash", False):
                status = "CRASHED"
            elif info.get("out_of_road", False):
                status = "OUT OF ROAD"
            else:
                status = "TIMEOUT"

        raw = env.unwrapped.render(
            mode="topdown",
            window=False,
            screen_size=screen_size,
            camera_position=env.agent.position,
            scaling=4.5,
        )
        frames.append(annotate_frame(raw, title, step, speed, last_action, rc, status))

    return frames, info.get("ep_metrics", {})


def combine_side_by_side(frames_left, frames_right, hold_end_frames=15):
    n = max(len(frames_left), len(frames_right)) + hold_end_frames
    combined = []
    sep = np.full((frames_left[0].shape[0], 6, 3), 240, dtype=np.uint8)
    for i in range(n):
        fl = frames_left[min(i, len(frames_left) - 1)]
        fr = frames_right[min(i, len(frames_right) - 1)]
        combined.append(np.concatenate([fl, sep, fr], axis=1))
    return combined


def save_video_and_gif(frames, mp4_path, gif_path=None, fps=10):
    Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    if gif_path:
        Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
        # Subsample every 2nd frame for compact, smooth GIFs
        pil_frames = [Image.fromarray(f) for f in frames[::2]]
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=180,
            loop=0,
            optimize=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="waymo")
    p.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--bc", default="runs/bc/actor.pt")
    p.add_argument("--out-dir", default="runs/videos")
    p.add_argument("--gif-dir", default=None)
    a = p.parse_args()

    bc_act = load_standalone_bc(a.bc)
    env = GateEnv(a.data, sequential=True)
    expert = LogTrackingExpert(env, speed_gain=0.5)

    for s in a.scenarios:
        f_exp, m_exp = rollout_frames(env, expert.act, s, f"Log-Tracking Expert ({a.data} #{s})")
        f_bc, m_bc = rollout_frames(env, bc_act, s, f"BC Policy B0 ({a.data} #{s})")
        both = combine_side_by_side(f_exp, f_bc)
        stem = f"{a.data}_scen{s}_expert_vs_bc"
        mp4_path = Path(a.out_dir) / f"{stem}.mp4"
        gif_path = (Path(a.gif_dir) / f"{stem}.gif") if a.gif_dir else (Path(a.out_dir) / f"{stem}.gif")
        save_video_and_gif(both, mp4_path, gif_path, fps=10)
        print(
            f"Saved {mp4_path} | Expert: rc={m_exp.get('route_completion', 0):.2f} succ={int(m_exp.get('success', 0))} "
            f"| BC: rc={m_bc.get('route_completion', 0):.2f} succ={int(m_bc.get('success', 0))} "
            f"crash={int(m_bc.get('crash', 0))} oor={int(m_bc.get('out_of_road', 0))}"
        )
    env.close()


if __name__ == "__main__":
    main()
