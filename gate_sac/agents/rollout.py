"""Imagined rollouts: fixed-horizon MBPO, the uncertainty gate, and the MOPO-style penalty.

One function covers every model-based method in the experiment matrix:

    gate=False, penalty=0   -> fixed-horizon MBPO baseline (B3)
    gate=False, penalty>0   -> penalty only (A1)
    gate=True,  penalty=0   -> gate only (A2)
    gate=True,  penalty>0   -> GATE-SAC (G)

Uncertainty is made unit-free with reference statistics of the *one-step* disagreement on real
start states (EMA over rounds):  u_norm = u / median_ref.  The gate truncates a rollout as soon as
u_norm exceeds the running p-th percentile of that reference: an adaptive threshold instead of a
hand-tuned constant. The penalty is r_tilde = r - lambda * u_norm.

Rollouts are converted to n-step targets (Dreamer-SAC eq. 20): each kept transition carries the
discounted reward/cost sum to the end of its (possibly truncated) rollout, the bootstrap state and
the bootstrap discount gamma^N * (1 - done).
"""
from dataclasses import dataclass

import torch


@dataclass
class RolloutConfig:
    horizon: int = 5
    batch_size: int = 5000  # number of real start states per round
    gate: bool = False
    gate_percentile: float = 80.0
    cost_aware_gate: bool = False  # keep predicted crashes even if uncertain (ablation)
    penalty: float = 0.0  # lambda in r - lambda * u_norm
    n_step: bool = True
    ref_ema: float = 0.9


class UncertaintyReference:
    """EMA of the median and p-th percentile of one-step disagreement on real states."""

    def __init__(self, percentile, ema):
        self.q, self.ema = percentile / 100.0, ema
        self.median = self.threshold = None

    def update(self, u_first):
        med = u_first.median().item()
        thr = torch.quantile(u_first, self.q).item() / max(med, 1e-12)
        if self.median is None:
            self.median, self.threshold = med, thr
        else:
            self.median = self.ema * self.median + (1 - self.ema) * med
            self.threshold = self.ema * self.threshold + (1 - self.ema) * thr


@torch.no_grad()
def imagine(ensemble, actor, start_obs, cfg: RolloutConfig, ref: UncertaintyReference, gamma):
    B, H = len(start_obs), cfg.horizon
    obs = start_obs
    alive = torch.ones(B, dtype=torch.bool, device=obs.device)
    steps = dict(obs=[], act=[], rew=[], cost=[], next_obs=[], done=[], kept=[], u=[])
    for k in range(H):
        act, _ = actor.action_log_prob(obs)
        next_obs, rew, cost_p, done_p, u = ensemble.step(obs, act)
        if k == 0:
            ref.update(u)
        u_norm = u / max(ref.median, 1e-12)
        keep = alive.clone()
        if cfg.gate:
            uncertain = u_norm > ref.threshold
            if cfg.cost_aware_gate:
                uncertain &= cost_p < 0.5
            keep &= ~uncertain
        done = (done_p > 0.5) | (cfg.cost_aware_gate & (cost_p >= 0.5))
        for key, val in zip(steps, [obs, act, rew - cfg.penalty * u_norm, cost_p, next_obs, done.float(), keep, u_norm]):
            steps[key].append(val)
        alive = keep & ~done
        obs = next_obs
        if not alive.any():
            break
    st = {k: torch.stack(v) for k, v in steps.items()}  # [K, B, ...]
    batch = _targets(st, gamma, cfg.n_step)

    kept = st["kept"]
    stats = dict(
        transitions=int(kept.sum()),
        mean_length=kept.float().sum(0).mean().item(),
        truncation_rate=1.0 - kept[-1].float().mean().item() if cfg.gate else 0.0,
        u_norm_mean=st["u"][kept].mean().item() if kept.any() else 0.0,
        threshold=ref.threshold,
        imagined_cost=st["cost"][kept].mean().item() if kept.any() else 0.0,
    )
    return batch, stats


def _targets(st, gamma, n_step):
    """Backward pass over each rollout to build (reward sum, cost sum, bootstrap obs, discount)."""
    R, C = st["rew"].clone(), st["cost"].clone()
    boot = st["next_obs"].clone()
    disc = gamma * (1.0 - st["done"])
    if n_step:
        for k in range(len(R) - 2, -1, -1):
            cont = st["kept"][k + 1]  # the rollout continued past step k
            R[k] = torch.where(cont, R[k] + gamma * R[k + 1], R[k])
            C[k] = torch.where(cont, C[k] + gamma * C[k + 1], C[k])
            disc[k] = torch.where(cont, gamma * disc[k + 1], disc[k])
            boot[k] = torch.where(cont[:, None], boot[k + 1], boot[k])
    m = st["kept"]
    return dict(obs=st["obs"][m], act=st["act"][m], rew=R[m], cost=C[m], boot_obs=boot[m], disc=disc[m])
