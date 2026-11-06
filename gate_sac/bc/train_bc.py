"""Behavior cloning (B0).

The BC policy *is* an SB3 SAC actor (tanh-squashed Gaussian, 3x256 MLP), so its weights load
straight into the RL agent. Loss = NLL of the expert action + MSE of the mean action.

    python -m gate_sac.bc.train_bc --demos runs/demos.npz --out runs/bc/actor.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.sac.policies import SACPolicy

ACTOR_NET_ARCH = [256, 256, 256]


def make_sac_policy(obs_dim, net_arch=None):
    obs_space = spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
    net_arch = net_arch or dict(pi=ACTOR_NET_ARCH, qf=[256, 256])
    return SACPolicy(obs_space, act_space, lambda _: 3e-4, net_arch=net_arch)


def load_bc_actor(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    actor = make_sac_policy(ckpt["obs_dim"], ckpt["net_arch"]).actor.to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.set_training_mode(False)
    return actor


def bc_log_prob(actor, obs, actions):
    mean, log_std, kwargs = actor.get_action_dist_params(obs)
    dist = actor.action_dist.proba_distribution(mean, log_std, **kwargs)
    return dist.log_prob(actions.clamp(-0.999, 0.999)), torch.tanh(mean)


def train_bc(demos, epochs=50, batch_size=256, lr=3e-4, val_frac=0.1, device="cuda", seed=0):
    torch.manual_seed(seed)
    obs = torch.as_tensor(demos["obs"], dtype=torch.float32)
    act = torch.as_tensor(demos["expert_action"], dtype=torch.float32)
    perm = torch.randperm(len(obs), generator=torch.Generator().manual_seed(seed))
    n_val = int(val_frac * len(obs))
    val, tr = perm[:n_val], perm[n_val:]

    policy = make_sac_policy(obs.shape[1])
    actor = policy.actor.to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    best, best_state = np.inf, None
    for epoch in range(epochs):
        actor.set_training_mode(True)
        for idx in tr[torch.randperm(len(tr))].split(batch_size):
            o, a = obs[idx].to(device), act[idx].to(device)
            logp, mean_a = bc_log_prob(actor, o, a)
            loss = -logp.mean() + F.mse_loss(mean_a, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
        actor.set_training_mode(False)
        with torch.no_grad():
            o, a = obs[val].to(device), act[val].to(device)
            val_mse = F.mse_loss(actor(o, deterministic=True), a).item()
        if val_mse < best:
            best, best_state = val_mse, {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  val action MSE {val_mse:.4f}")
    return dict(actor=best_state, obs_dim=obs.shape[1], net_arch=dict(pi=ACTOR_NET_ARCH, qf=[256, 256]), val_mse=best)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    ckpt = train_bc(dict(np.load(a.demos)), epochs=a.epochs, seed=a.seed,
                    device="cuda" if torch.cuda.is_available() else "cpu")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, a.out)
    print(f"best val MSE {ckpt['val_mse']:.4f} -> {a.out}")
