# RL Experiments

This repository is a workspace for running and tracking reinforcement learning (RL) experiments. It contains small prototypes, training scripts, and plots used to explore algorithms and environments. Everything is implemented from scratch.

## Experiments
- **CartPole**: REINFORCE (Monte-Carlo policy gradient) with per-episode updates and training curves in `cartpole/`.
- **LunarLanderContinuous**: PPO (actor-critic) with GAE and clipped objective, with metrics plots in `lunar_lander/`.

## Structure
All experiments share a common structure:
- `README.md`: description of the experiment
- `train.py`: training code and experiment-specific assets
- `figures/`: generated figures and plots

## Notes
This is an evolving sandbox; scripts and results may change as experiments progress.
