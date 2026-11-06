"""Ensemble of E probabilistic MLP dynamics models (MBPO-style), vectorized over members.

Each member maps (s, a) -> Gaussian over (delta s, r), plus logits for cost and termination.
Members see different bootstrap resamples of the real buffer, so their disagreement on the
next-state mean is an estimate of epistemic uncertainty:

    u(s, a) = mean_j  Var_i[ mu_i(s, a)_j ]      (j over normalized state dims, i over members)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):

    def __init__(self, n_members, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_members, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_members, 1, d_out))
        bound = 1.0 / np.sqrt(d_in)  # nn.Linear's default init
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):  # x: [E, N, d_in]
        return torch.baddbmm(self.bias, x, self.weight)


class Ensemble(nn.Module):

    def __init__(self, obs_dim, act_dim, n_members=4, hidden=256, n_layers=4):
        super().__init__()
        self.obs_dim, self.act_dim, self.n_members = obs_dim, act_dim, n_members
        dims = [obs_dim + act_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(EnsembleLinear(n_members, i, o) for i, o in zip(dims[:-1], dims[1:]))
        self.n_gauss = obs_dim + 1  # delta s, r
        self.head = EnsembleLinear(n_members, hidden, 2 * self.n_gauss + 2)  # mu, logvar, cost, done
        self.max_logvar = nn.Parameter(torch.full((1, 1, self.n_gauss), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, self.n_gauss), -10.0))
        for name in ["in_mu", "in_std", "out_mu", "out_std"]:
            self.register_buffer(name, torch.zeros(1))

    def _set_norm(self, x, y):
        self.in_mu, self.in_std = x.mean(0), x.std(0).clamp_min(1e-6)
        self.out_mu, self.out_std = y.mean(0), y.std(0).clamp_min(1e-6)

    def _forward(self, x):  # x normalized, [E, N, d]
        h = x
        for layer in self.layers:
            h = F.silu(layer(h))
        out = self.head(h)
        mu, logvar = out[..., :self.n_gauss], out[..., self.n_gauss:2 * self.n_gauss]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mu, logvar, out[..., -2], out[..., -1]

    def _inputs(self, obs, act):
        return (torch.cat([obs, act], -1) - self.in_mu) / self.in_std

    # ------------------------------------------------------------------ training
    def fit(self, data, optimizer, batch_size=256, holdout_frac=0.1, max_epochs=50, max_updates=None,
            patience=5, max_holdout=5000):
        """Fit on a dict of float tensors (obs, act, rew, cost, next_obs, done). Returns holdout stats."""
        device = next(self.parameters()).device
        x = torch.cat([data["obs"], data["act"]], -1).to(device)
        y = torch.cat([data["next_obs"] - data["obs"], data["rew"][:, None]], -1).to(device)
        cost, done = data["cost"].to(device), data["done"].to(device)
        n = len(x)
        perm = torch.randperm(n, device=device)
        n_hold = min(int(holdout_frac * n), max_holdout)
        hold, train = perm[:n_hold], perm[n_hold:]
        self._set_norm(x[train], y[train])
        xn, yn = (x - self.in_mu) / self.in_std, (y - self.out_mu) / self.out_std
        # Rare events (crash / out-of-road) get up-weighted in the BCE losses.
        pos_w = lambda t: ((1 - t[train]).sum() / t[train].sum().clamp_min(1)).clamp(1, 100)
        cost_pw, done_pw = pos_w(cost), pos_w(done)

        boot = train[torch.randint(len(train), (self.n_members, len(train)), device=device)]
        best, best_state, bad, updates = np.inf, None, 0, 0
        for epoch in range(max_epochs):
            order = boot[:, torch.randperm(len(train), device=device)]
            for start in range(0, len(train), batch_size):
                idx = order[:, start:start + batch_size]
                mu, logvar, cost_logit, done_logit = self._forward(xn[idx])
                nll = (((mu - yn[idx]) ** 2) * torch.exp(-logvar) + logvar).mean()
                bce = F.binary_cross_entropy_with_logits(cost_logit, cost[idx], pos_weight=cost_pw) + \
                    F.binary_cross_entropy_with_logits(done_logit, done[idx], pos_weight=done_pw)
                loss = nll + bce + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
            hold_mse = self._holdout_mse(xn[hold], yn[hold]) if n_hold else 0.0
            if hold_mse < best - 1e-4:
                best, bad = hold_mse, 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                bad += 1
            if bad >= patience or (max_updates and updates >= max_updates):
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        return dict(holdout_mse=best, epochs=epoch + 1, updates=updates)

    @torch.no_grad()
    def _holdout_mse(self, xn, yn):
        mu, _, _, _ = self._forward(xn.expand(self.n_members, -1, -1))
        return ((mu - yn) ** 2).mean().item()

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def step(self, obs, act, sample=True):
        """One imagined step for a batch. Uses a random member per row (MBPO).

        Returns next_obs, reward, cost prob, done prob, and the disagreement u.
        """
        x = self._inputs(obs, act).expand(self.n_members, -1, -1)
        mu, logvar, cost_logit, done_logit = self._forward(x)
        u = mu[..., :self.obs_dim].var(0).mean(-1)  # normalized delta-s units
        member = torch.randint(self.n_members, (len(obs),), device=obs.device)
        rows = torch.arange(len(obs), device=obs.device)
        y = mu[member, rows]
        if sample:
            y = y + torch.randn_like(y) * torch.exp(0.5 * logvar[member, rows])
        y = y * self.out_std + self.out_mu
        next_obs = (obs + y[:, :self.obs_dim]).clamp(0.0, 1.0)
        return next_obs, y[:, -1], torch.sigmoid(cost_logit).mean(0), torch.sigmoid(done_logit).mean(0), u
