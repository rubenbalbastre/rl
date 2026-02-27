import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import os
import numpy as np


np.random.seed(0)
torch.manual_seed(0)

# 1. Define a simple policy network for the CartPole environment

class Policy(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

policy_model = Policy(input_dim=4, output_dim=2)

# 2. Training Algorithm

num_episodes = 1000
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-3)
gamma = 0.99

# 3. Environment setup

env = gym.make("CartPole-v1")
observations, info = env.reset()

# 4. Training loop

def discounted_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)

rewards_theshold = 450
losses = []
completed_rewards = []

for episode in range(num_episodes):

    observation, _ = env.reset()
    log_probs_episode = []
    rewards_episode = []
    done = False

    while not done:

        # get action from policy
        torch_observations = torch.tensor(observations, dtype=torch.float32)
        logits = policy_model(torch_observations)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        # get observation from environment
        observations, rewards, terminated, truncated, info = env.step(actions.numpy())
        done = terminated or truncated

        # store log probs and rewards
        log_probs_episode.append(dist.log_prob(actions))
        rewards_episode.append(rewards)
    
    # compute returns and advantages
    returns = discounted_returns(rewards_episode, gamma=gamma)
    advantages = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

    # loss
    loss = -(torch.stack(log_probs_episode) * advantages).sum()
    losses.append(loss.item())

    # update policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # logging
    completed_rewards.append(sum(rewards_episode))

    if len(completed_rewards) >= 100:
        avg100 = sum(completed_rewards[-100:]) / 100
        if avg100 >= rewards_theshold:
            print(f"Solved at step {episode}, avg100={avg100:.1f}")
            break


env.close()

# 5. Plotting results

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "lines.linewidth": 1.8,
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=120)

# Loss plot
ax = axes[0]
ax.plot(losses, color="#1f77b4")
ax.set_title("Policy Loss")
ax.set_xlabel("Updates")
ax.set_ylabel("Loss")
ax.grid(True)

# Return plot
ax = axes[1]
ax.plot(completed_rewards, color="#2ca02c", alpha=0.5, label="Episode return")
window = min(100, len(completed_rewards))
if window >= 2:
    rolling = np.convolve(completed_rewards, np.ones(window) / window, mode="valid")
    rolling_steps = list(range(window - 1, len(completed_rewards)))
    ax.plot(rolling_steps, rolling, color="#2ca02c", label=f"{window}-episode mean")
ax.set_title("Episode Returns")
ax.set_xlabel("Episodes")
ax.set_ylabel("Return")
ax.grid(True)
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

figures_dir = "experiments/cartpole/figures/"
os.makedirs(figures_dir, exist_ok=True)
fig.savefig(f"{figures_dir}cartpole_training.png")
