# CartPole Experiment

This folder contains a simple CartPole-v1 experiment using a policy-gradient (REINFORCE) style trainer.

## Files
- `train.py`: training script and plotting
- `figures/`: output figures (e.g., training curves)

## Training Loop
The training loop collects experience from environments, stores per-episode trajectories, and updates the policy when episodes complete. At each environment step:
- The policy produces action logits from the current observations.
- Actions are sampled from a categorical distribution.
- Rewards and actions are stored per environment until the episode ends.

When an episode finishes, the script:
- Computes discounted returns for that episode.
- Normalizes returns to reduce variance.
- Computes the REINFORCE loss and performs one optimizer step.

## RL Algorithm
This experiment uses vanilla REINFORCE (Monte-Carlo policy gradient):
1. Roll out a full episode.
2. Compute discounted returns \(G_t\).
3. Update the policy by maximizing \(\sum_t \log \pi(a_t|s_t) \cdot G_t\).

This is an on-policy method; it updates using data collected by the current policy and performs updates only after full episodes complete.

## Run
```bash
python train.py
```

## Output
The script saves a plot to `experiments/cartpole/figures/cartpole_training.png`.
