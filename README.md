# RL Experiments

This repository is a workspace for running and tracking reinforcement learning (RL) experiments. It contains small prototypes, training scripts, and plots used to explore algorithms and environments.

## Structure
- `experiments/`: training code and experiment-specific assets
- `experiments/*/figures/`: generated figures and plots

## Experiments
- **CartPole-v1**: REINFORCE (Monte-Carlo policy gradient) with per-episode updates and training curves in `experiments/cartpole/`.
- **LunarLanderContinuous-v2**: PPO (actor-critic) with GAE and clipped objective, with metrics plots in `experiments/lunar_lander/`.

## Notes
This is an evolving sandbox; scripts and results may change as experiments progress.
