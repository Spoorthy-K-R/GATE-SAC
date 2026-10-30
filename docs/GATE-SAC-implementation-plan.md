# GATE-SAC: Feasibility Review & Implementation Plan

## TL;DR

- **Verdict: feasible within the semester**, *if* you narrow the scope. The core idea (gate + pessimistic penalty + Lagrangian SAC on a vector-state ensemble) is cheap to compute and the basic signals exist.
- **Biggest risks** are in the parts *around* the core idea: Waymo/ScenarioNet for BC, a contradictory BC↔SAC initialization, and an uncertainty scale that makes fixed thresholds / λ meaningless.
- I ran an empirical feasibility test on your M2. Script: [feasibility_test.py](file:///Users/spoorthy/Documents/SEM%20III/DRL/GATE-SAC-DRL-project/feasibility/feasibility_test.py). Raw results: [feasibility_report.json](file:///Users/spoorthy/Documents/SEM%20III/DRL/GATE-SAC-DRL-project/feasibility/results/feasibility_report.json).

---

## 1. Dreamer-SAC: What to Borrow

| Dreamer-SAC finding | Implication for GATE-SAC |
|---|---|
| Inverted-U in horizon; best at **H=5** | Your fixed-H MBPO baseline must sweep H ∈ {1,3,5,10} and compare to the *best* one (your proposal already says this — keep it). |
| Real + imagined mix beats imagined-only (A1 vs A2, A3 vs A4) | Always train SAC on a real/imagined mix; report the ratio. |
| **n-step targets** on imagined rollouts beat 1-step (A3 vs A7) | Cheap win — add it. Note: gating truncates rollouts, so N = *surviving* length (naturally compatible). |
| Imagined buffer **discarded each iteration**, rollouts start from real states | Do the same; avoids training on stale model predictions. |
| Model **over-predicts** rare crash events (Sec 4.5) | Directly relevant to your gate-near-crashes concern (see §3, Issue 4). |
| Future work: *"adaptive rollout strategies by model reliability and uncertainty"* | This is exactly GATE-SAC's pitch — cite it as motivation. |

> [!NOTE]
> Dreamer-SAC uses images + RSSM. GATE-SAC uses vectors + MLP ensemble (MBPO-style), so it is **not** a direct comparison. Frame it as "uncertainty-adaptive horizon in state space", with Dreamer-SAC as motivation, not as a baseline.

---

## 2. Feasibility Test Results (Measured on Your M2)

Setup: MetaDrive 0.4.3, 3-block procedural maps, traffic 0.15, built-in PPO expert + Gaussian action noise (σ=0.25), 8k train / 3k held-out steps on **unseen** scenarios, 4-member probabilistic ensemble.

![Feasibility plots](/Users/spoorthy/.gemini/antigravity-ide/brain/56b2e402-8717-4a78-9760-4485cc93592d/feasibility_plots.png)

| Test | Result | Meaning |
|---|---|---|
| T1 Sim throughput | **~170–200 steps/s** (1 core) | 100k env steps ≈ 10 min of sim. Local dev is fine; HPRC only for multi-seed sweeps. |
| T2 Expert for BC | Built-in PPO expert works out of the box (success ~33%, return 335 on hard maps) | **You don't need Waymo for BC.** An imperfect expert also leaves room for RL to improve on it. |
| T3 Model error vs H | Normalized MSE: 0.26 (h=1) → 0.92 (h=5) → **1.0 at h≈6** | Past ~5 steps the model is no better than predicting the mean. Useful horizon is **3–5**, matching Dreamer-SAC. |
| T4 Gate validity | Spearman(u, true error) ≈ **0.51–0.54**; dropping top 10–20% by u removes transitions with **4–5× higher error** | ✅ The gate's core assumption holds, but the signal is moderate and noisy. |
| T4 Uncertainty scale | Mean u ≈ **6e-6 to 1e-5** | ❌ A fixed threshold like `0.08` **never fires**. That's why your earlier Colab prototype showed 0% truncation. `λ·u` with λ=0.5 has basically no effect. |
| T5 Pre-crash signal | u is **1.86× higher** in the 5 steps before a crash/out-of-road | ⚠️ This confirms your proposal's own worry: **the gate will cut exactly the near-crash states.** |
| T5 Cost prediction | Crash/out-of-road AUC = **0.78** (only 26 events in training) | Imagined costs are weak. Don't trust the Lagrangian on imagined costs alone. |
| T6 Imagination speed | ~37k imagined transitions/s (CPU) | Model rollouts won't be the bottleneck. |

> [!WARNING]
> **Environment gotcha:** PyPI `metadrive-simulator` 0.4.x needs **Python < 3.12**. I created a conda env at `GATE-SAC-DRL-project/.venv` (Python 3.11) with metadrive, torch, scipy, sklearn, and matplotlib. Pin this on HPRC too.

---

## 3. Issues in the Proposal and Recommended Fixes

### Issue 1: BC init contradicts "randomly initialized SAC" — 🔴 High
The proposal says it will *fine-tune an imitation policy*, but also says *"a standard, randomly initialized SAC agent"*. If you copy BC weights into the actor and pair them with a random critic, the actor usually **unlearns** BC in the first few thousand updates. Your prototype loaded `bc.net[:4]` into the actor trunk this way.

**Fix:** pick one of these and say so in the report:
1. **BC-init + critic warm-up (recommended):** freeze the actor for ~5k updates while Q and Q_c learn on BC rollouts, then unfreeze. Add a BC regularizer `β·||π(s) − a_expert||²` and anneal β → 0.
2. Random init, with BC data only seeding the replay buffer (simpler, but then it's no longer "fine-tuning").

### Issue 2: Waymo / ScenarioNet for BC — 🔴 High
- Waymo logs contain **trajectories, not steering/throttle**. You would need inverse dynamics to get action labels for BC.
- In ScenarioEnv, log-replayed traffic is **non-reactive**, which weakens the "real human unpredictability" claim.
- Getting a license, downloading, and converting with ScenarioNet could eat 1–2 weeks.

**Fix:** use MetaDrive's built-in PPO expert (or IDM) in **SafeMetaDriveEnv** for the core results. Move Waymo/ScenarioNet to a stretch goal, e.g. "evaluate on MetaDrive's bundled real-world scenarios".

### Issue 3: Where do the "safety-critical failure scenarios" come from? — 🟠 Medium
They're never defined. **Fix:** after training BC, evaluate it on ~500 seeds, collect the ones where it fails, and fine-tune on that **failure set** (plus some normal seeds so it doesn't forget). This makes the "RL fixes IL failures" story concrete and measurable. Use `SafeMetaDriveEnv`: it has static obstacles and a built-in `cost` signal.

### Issue 4: The gate removes near-crash states — 🟠 Medium (confirmed by T5)
**Fix (the novelty angle):** a **cost-aware gate**. If the ensemble's mean cost prediction is high, **keep the transition as a terminal with cost=1** instead of dropping it. Only drop high-u transitions with low predicted cost. Report the truncation rate near crashes vs elsewhere.

### Issue 5: Scale of u, λ, and threshold — 🟠 Medium
- **Normalize u**, e.g. `u / running_median(u)`, or use MOPO's `max_i ||σ_i||`. Then λ and the threshold are unit-free.
- Use the **running-percentile gate** your related-work section already promises (e.g. truncate above the 80th percentile, computed per batch or as an EMA).
- Sweep λ ∈ {0, 0.1, 0.5, 1} on the *normalized* u.

### Issue 6: Crash penalty is counted twice — 🟠 Medium
Your reward has `−λ_c·c_collision` **and** a separate cost critic. In SAC-Lagrangian, the reward **should not** include the crash penalty. Use `r = progress − lane_dev`, and apply cost only through `Q_c` and λ_c. Update λ_c from **real episode cost**, converted to the per-km budget (`cost_per_km = Σc / km_driven`), not from the mean of `Q_c`.

### Issue 7: Baselines and evaluation are too thin — 🟡 Medium
Only one comparison (vs MBPO) can't tell you **which part** helped. See the experiment matrix in §5. Use ≥3 seeds (5 if HPRC allows) and evaluate on held-out seeds.

### Issue 8: The SB3 dependency — 🟡 Low
SB3's SAC doesn't easily support two replay buffers, n-step imagined targets, a cost critic, or λ updates. **Fix:** write your own ~400-line CleanRL-style SAC/SAC-Lag. Use SB3's SAC only as a sanity check for the model-free baseline.

---

## 4. Revised GATE-SAC Algorithm

```
Init: BC actor π (trained on expert data), critics Q1,Q2, cost critic Qc, λ_c, ensemble {f_i}, D_real ← expert data
Warm-up: train Q, Qc with π frozen (≈5k updates)
loop every env step:
    act in SafeMetaDriveEnv with π → store (s,a,r,c,s',d) in D_real
    every 250 steps: fit ensemble on D_real (holdout early-stop)
    every 250 steps: D_img ← ∅; sample B real states
        for k in 1..H_max (=5):
            a ~ π(s); (s', r̂, ĉ, d̂), u = ensemble(s,a);  ũ = u / median(u)
            if ũ > percentile_p(ũ) and ĉ < 0.5: stop this rollout   # cost-aware gate
            if ĉ ≥ 0.5: store terminal (s,a,r̂-λũ,1,s',done=1); stop
            store (s,a, r̂ - λ·ũ, ĉ, s', d̂) in D_img
    G gradient steps: batch = 5% real + 95% imagined (MBPO) — sweep {0.05, 0.5}
        Q targets: 1-step on real, n-step on imagined (Dreamer-SAC eq. 20)
        Qc target: same, with cost
        actor loss: α logπ − Q + λ_c · Qc  (+ β·BC loss, anneal β)
    every episode: λ_c ← max(0, λ_c + η (cost_per_km − budget))
```

---

## 5. Experiment Matrix

| ID | Method | Model | Gate | Penalty | Lagrangian | BC init | Purpose |
|---|---|---|---|---|---|---|---|
| B0 | BC only | – | – | – | – | ✓ | IL floor |
| B1 | SAC | – | – | – | – | ✓ | Model-free floor |
| B2 | SAC-Lag | – | – | – | ✓ | ✓ | Isolates safety layer |
| B3 | MBPO (H=1,3,5,10) | ✓ | – | – | ✓ | ✓ | Fixed-H baseline (pick best) |
| A1 | + penalty only (MOPO) | ✓ | – | ✓ | ✓ | ✓ | Ablation |
| A2 | + gate only | ✓ | ✓ | – | ✓ | ✓ | Ablation |
| A3 | + gate (cost-unaware) + penalty | ✓ | ✓ | ✓ | ✓ | ✓ | Shows value of cost-aware gate |
| **G** | **GATE-SAC (full)** | ✓ | ✓ cost-aware | ✓ | ✓ | ✓ | Proposed |

**Metrics** (held-out seeds, both the failure set and normal seeds): success rate, route completion, collisions/km, out-of-road/km, return, **env steps until X% success** (sample efficiency), truncation rate (overall vs near crashes), model error vs h, and λ_c over time.

**Compute:** ~10 runs × 3–5 seeds × 100k steps. At roughly 1–2 h per run on CPU, that's about 50–100 CPU-hours. That's comfortable on HPRC as a job array, or feasible locally with 3 seeds overnight.

---

## 6. Timeline (Oct 22 → ~Dec 9)

> [!IMPORTANT]
> This assumes the final report is due around Dec 8–9. Shift the weeks if your deadline is different.

| Week | Dates | Deliverables | Exit criterion |
|---|---|---|---|
| **1** | Oct 22–28 | Repo structure, `.venv`/HPRC env, SafeMetaDriveEnv wrapper with cost + km logging, eval harness (held-out seeds, per-km metrics), TensorBoard/W&B. Collect ~50k expert transitions; train **BC (B0)**; mine the **BC failure seed set** | BC numbers on 100 held-out seeds + failure set identified |
| **2** | Oct 29–Nov 4 | Own SAC + SAC-Lag; BC-init + critic warm-up + annealed BC loss; run **B1, B2** (1 seed, 100k steps); sanity-check against SB3 SAC | SAC improves on BC on the failure set without collapsing |
| **3** | Nov 5–11 | Ensemble (probabilistic, holdout early stopping, u normalization); model diagnostics (reuse feasibility script); **MBPO B3** with an H sweep | MBPO beats SAC in sample efficiency at some H |
| **4** | Nov 12–18 | Percentile gate, MOPO penalty, **cost-aware gate**, n-step imagined targets → **GATE-SAC**; tune λ and percentile on 1 seed. *(Likely progress-report week: present B0–B3 + first G curves)* | G runs stably; truncation rate in 5–30% range |
| **5** | Nov 19–25 | Full sweep on HPRC: all configs × 3–5 seeds; ablations A1–A3 *(Thanksgiving week — launch jobs before the break)* | All runs finished |
| **6** | Nov 26–Dec 2 | Analysis: learning curves with CIs, safety table, truncation-near-crash analysis, λ_c dynamics. Optional stretch: bundled real-world (Waymo/nuScenes) scenarios for eval only | Figures + tables final |
| **7** | Dec 3–9 | Report, video of rollouts, code cleanup, buffer for reruns | Submit |

**Decision points:**
- **End of week 2:** if SAC-from-BC collapses → switch to option 2 in Issue 1 (random init + BC data in the buffer).
- **End of week 3:** if MBPO is no better than SAC → shorten H to 1–3 and raise the real-data ratio. Gating still matters, but frame it as a *safety* improvement rather than a sample-efficiency one.

---

## 7. Lessons from Your Earlier Colab Prototype

- The "expert" `steering = -obs[1]*1.5` isn't a real expert. Use `metadrive.examples.ppo_expert.numpy_expert.expert`.
- `uncertainty_threshold=0.08` was 4 orders of magnitude above the real u (~1e-5), so the gate never fired.
- The `λ_c` clamp to 0.3 is a symptom of Issue 6 (double-counted crash penalty) plus no critic warm-up.
- "Zero collisions" came from a car standing still. Always report success / route completion **next to** collisions.
- 5k steps and 1 seed are too few. Use ≥50k–100k steps and ≥3 seeds.
