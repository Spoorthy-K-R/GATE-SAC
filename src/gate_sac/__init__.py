"""GATE-SAC: Uncertainty-Gated Reinforcement Fine-Tuning for Safety-Critical Autonomous Driving.

Package layout (see docs/ARCHITECTURE.md for the full design):

- ``gate_sac.config``       typed experiment configuration (dataclasses + YAML)
- ``gate_sac.envs``         MetaDrive / SafeMetaDrive factory, metric + seed wrappers, dummy env
- ``gate_sac.data``         replay buffers, transition I/O, data collection
- ``gate_sac.models``       actor/critic networks, probabilistic world-model ensemble
- ``gate_sac.agents``       SAC / SAC-Lagrangian, Lagrange multiplier, behavior cloning
- ``gate_sac.imagination``  uncertainty calibration, gating, branched model rollouts
- ``gate_sac.training``     training loop orchestration and evaluation
- ``gate_sac.utils``        logging, seeding, torch helpers
"""

__version__ = "0.1.0"
