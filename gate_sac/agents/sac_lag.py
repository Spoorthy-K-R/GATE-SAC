"""SAC-Lagrangian with optional model-based imagination: the GATE-SAC learner.

Subclasses SB3's SAC and overrides `train()`; SB3 still provides the actor, twin critics, entropy
tuning, environment loop, logging and checkpointing. Added on top:

* cost critic Q_c (twin, pessimistic max) trained on the cost signal c
* Lagrange multiplier lambda_c, updated by dual ascent on *real* collisions-per-km vs a budget
* actor loss  alpha*log pi - min Q + lambda_c * Q_c + beta * ||pi(s) - pi_BC(s)||^2  (beta annealed)
* BC warm start: actor initialized from BC and frozen for `warmup_updates` while critics learn
* optional world-model ensemble + imagined rollouts (gate / penalty / fixed-H), mixed into every
  batch at `real_ratio` real : (1 - real_ratio) imagined; real data uses 1-step targets,
  imagined data n-step targets.
"""
from collections import deque

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update

from gate_sac.agents.rollout import RolloutConfig, UncertaintyReference, imagine
from gate_sac.models.ensemble import Ensemble


class CostReplayBuffer(ReplayBuffer):
    """SB3 replay buffer that also stores the per-step cost from `info["cost"]`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        self.costs[self.pos] = np.array([info.get("cost", 0.0) for info in infos])
        super().add(obs, next_obs, action, reward, done, infos)

    def size_(self):
        return self.buffer_size if self.full else self.pos

    def all_data(self, device):
        """Every stored real transition (n_envs = 1), for fitting the world model."""
        n = self.size_()
        t = lambda x: th.as_tensor(x[:n, 0], dtype=th.float32, device=device)
        done = self.dones[:n, 0] * (1 - self.timeouts[:n, 0])
        return dict(obs=t(self.observations), act=t(self.actions), rew=t(self.rewards), cost=t(self.costs),
                    next_obs=t(self.next_observations), done=th.as_tensor(done, dtype=th.float32, device=device))

    def sample_lag(self, batch_size, gamma, device):
        idx = np.random.randint(0, self.size_(), size=batch_size)
        t = lambda x: th.as_tensor(x[idx, 0], dtype=th.float32, device=device)
        done = self.dones[idx, 0] * (1 - self.timeouts[idx, 0])
        return dict(obs=t(self.observations), act=t(self.actions), rew=t(self.rewards), cost=t(self.costs),
                    boot_obs=t(self.next_observations),
                    disc=th.as_tensor(gamma * (1 - done), dtype=th.float32, device=device))


class SACLag(SAC):

    def __init__(
        self,
        policy,
        env,
        *,
        # safety layer
        lagrangian=True,
        cost_budget=1.0,  # collisions + out-of-road per km
        lambda_init=0.0,
        lambda_lr=0.01,
        lambda_max=100.0,
        cost_window=20,  # episodes in the running cost-per-km estimate
        # BC fine-tuning
        bc_actor_path=None,
        warmup_updates=5000,
        bc_reg=1.0,
        bc_reg_anneal_updates=50_000,
        # model-based
        use_model=False,
        n_members=4,
        model_train_freq=250,
        model_max_updates=5000,
        real_ratio=0.05,
        rollout=None,
        **kwargs,
    ):
        kwargs.setdefault("replay_buffer_class", CostReplayBuffer)
        self.lagrangian, self.cost_budget, self.lambda_c = lagrangian, cost_budget, lambda_init
        self.lambda_lr, self.lambda_max, self.cost_window = lambda_lr, lambda_max, cost_window
        self.bc_actor_path, self.warmup_updates = bc_actor_path, warmup_updates
        self.bc_reg, self.bc_reg_anneal_updates = bc_reg, bc_reg_anneal_updates
        self.use_model, self.n_members = use_model, n_members
        self.model_train_freq, self.model_max_updates, self.real_ratio = model_train_freq, model_max_updates, real_ratio
        self.rollout_cfg = RolloutConfig(**(rollout or {}))
        super().__init__(policy, env, **kwargs)

    # ------------------------------------------------------------------ setup
    def _setup_model(self):
        super()._setup_model()
        make_critic = lambda: self.policy.make_critic(features_extractor=None).to(self.device)
        self.cost_critic, self.cost_critic_target = make_critic(), make_critic()
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_target.set_training_mode(False)
        self.cost_critic.optimizer = th.optim.Adam(self.cost_critic.parameters(), lr=self.lr_schedule(1))

        self.bc_actor = None
        if self.bc_actor_path:
            from gate_sac.bc.train_bc import load_bc_actor
            bc_actor = load_bc_actor(self.bc_actor_path, self.device)
            self.actor.load_state_dict(bc_actor.state_dict())
            self.bc_actor = bc_actor

        self.ensemble = None
        if self.use_model:
            self.ensemble = Ensemble(self.observation_space.shape[0], self.action_space.shape[0],
                                     n_members=self.n_members).to(self.device)
            self.ensemble_opt = th.optim.Adam(self.ensemble.parameters(), lr=1e-3, weight_decay=1e-5)
            self.unc_ref = UncertaintyReference(self.rollout_cfg.gate_percentile, self.rollout_cfg.ref_ema)
        self.imagined = None
        self._last_model_step = -np.inf
        self._episodes = deque(maxlen=self.cost_window)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "cost_critic", "cost_critic_target", "bc_actor", "ensemble", "ensemble_opt", "unc_ref", "imagined",
            "_episodes", "bc_actor_path"]

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        return state_dicts + ["cost_critic", "cost_critic_target", "cost_critic.optimizer"], variables

    def load_demos(self, demos):
        """Seed the real buffer with demonstration transitions (dict of arrays from bc.collect)."""
        for i in range(len(demos["obs"])):
            info = dict(cost=float(demos["cost"][i]), **{"TimeLimit.truncated": bool(demos["truncated"][i])})
            done = bool(demos["terminated"][i] or demos["truncated"][i])
            self.replay_buffer.add(demos["obs"][i:i + 1], demos["next_obs"][i:i + 1], demos["action"][i:i + 1],
                                   demos["reward"][i:i + 1], np.array([done]), [info])

    # ------------------------------------------------------- lagrangian (real data)
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        for info in infos:
            ep = info.get("ep_metrics")
            if ep is None:
                continue
            self._episodes.append(ep)
            for k in ["ret", "cost", "success", "crash", "out_of_road", "route_completion", "avg_speed_kmh"]:
                self.logger.record_mean(f"episode/{k}", ep[k])
            km = sum(e["km"] for e in self._episodes)
            cost_per_km = sum(e["cost"] for e in self._episodes) / max(km, 1e-6)
            self.logger.record("episode/cost_per_km_window", cost_per_km)
            if self.lagrangian:
                self.lambda_c = float(np.clip(self.lambda_c + self.lambda_lr * (cost_per_km - self.cost_budget),
                                              0.0, self.lambda_max))

    # ------------------------------------------------------------ world model
    def _refresh_imagination(self):
        data = self.replay_buffer.all_data(self.device)
        fit = self.ensemble.fit(data, self.ensemble_opt, max_updates=self.model_max_updates)
        n = len(data["obs"])
        start = data["obs"][th.randint(n, (self.rollout_cfg.batch_size,), device=self.device)]
        self.imagined, stats = imagine(self.ensemble, self.actor, start, self.rollout_cfg, self.unc_ref, self.gamma)
        for k, v in {**fit, **stats}.items():
            self.logger.record(f"model/{k}", v)
        self._last_model_step = self.num_timesteps

    def _sample_batch(self, batch_size):
        if self.imagined is None or len(self.imagined["obs"]) == 0:
            return self.replay_buffer.sample_lag(batch_size, self.gamma, self.device)
        n_real = max(1, int(round(self.real_ratio * batch_size)))
        real = self.replay_buffer.sample_lag(n_real, self.gamma, self.device)
        idx = th.randint(len(self.imagined["obs"]), (batch_size - n_real,), device=self.device)
        return {k: th.cat([real[k], self.imagined[k][idx]]) for k in real}

    # ------------------------------------------------------------------ train
    def train(self, gradient_steps, batch_size=64):
        if self.use_model and self.num_timesteps - self._last_model_step >= self.model_train_freq:
            self._refresh_imagination()

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.cost_critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        logs = {k: [] for k in ["critic_loss", "cost_critic_loss", "actor_loss", "ent_coef", "q", "qc", "bc_loss"]}
        for _ in range(gradient_steps):
            b = self._sample_batch(batch_size)
            rew, cost, disc = b["rew"][:, None], b["cost"][:, None], b["disc"][:, None]
            actor_frozen = self._n_updates < self.warmup_updates

            actions_pi, log_prob = self.actor.action_log_prob(b["obs"])
            log_prob = log_prob.reshape(-1, 1)
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                if not actor_frozen:
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor

            with th.no_grad():
                next_a, next_logp = self.actor.action_log_prob(b["boot_obs"])
                next_q = th.min(th.cat(self.critic_target(b["boot_obs"], next_a), 1), 1, keepdim=True)[0]
                target_q = rew + disc * (next_q - ent_coef * next_logp.reshape(-1, 1))
                next_qc = th.max(th.cat(self.cost_critic_target(b["boot_obs"], next_a), 1), 1, keepdim=True)[0]
                target_qc = cost + disc * next_qc

            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in self.critic(b["obs"], b["act"]))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            cost_critic_loss = 0.5 * sum(F.mse_loss(q, target_qc) for q in self.cost_critic(b["obs"], b["act"]))
            self.cost_critic.optimizer.zero_grad()
            cost_critic_loss.backward()
            self.cost_critic.optimizer.step()

            if not actor_frozen:
                q_pi = th.min(th.cat(self.critic(b["obs"], actions_pi), 1), 1, keepdim=True)[0]
                qc_pi = th.max(th.cat(self.cost_critic(b["obs"], actions_pi), 1), 1, keepdim=True)[0]
                actor_loss = (ent_coef * log_prob - q_pi + self.lambda_c * qc_pi).mean()
                beta = self._bc_beta()
                if beta > 0 and self.bc_actor is not None:
                    with th.no_grad():
                        a_bc = self.bc_actor(b["obs"], deterministic=True)
                    bc_loss = F.mse_loss(self.actor(b["obs"], deterministic=True), a_bc)
                    actor_loss = actor_loss + beta * bc_loss
                    logs["bc_loss"].append(bc_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                logs["actor_loss"].append(actor_loss.item())
                logs["q"].append(q_pi.mean().item())
                logs["qc"].append(qc_pi.mean().item())

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.cost_critic.parameters(), self.cost_critic_target.parameters(), self.tau)
            logs["critic_loss"].append(critic_loss.item())
            logs["cost_critic_loss"].append(cost_critic_loss.item())
            logs["ent_coef"].append(ent_coef.item())
            self._n_updates += 1

        for k, v in logs.items():
            if v:
                self.logger.record(f"train/{k}", np.mean(v))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/lambda_c", self.lambda_c)
        self.logger.record("train/bc_beta", self._bc_beta())

    def _bc_beta(self):
        if self.bc_reg_anneal_updates <= 0:
            return 0.0
        progress = max(0, self._n_updates - self.warmup_updates) / self.bc_reg_anneal_updates
        return self.bc_reg * max(0.0, 1.0 - progress)
