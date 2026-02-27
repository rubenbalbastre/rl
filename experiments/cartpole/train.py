import gymnasium as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np


np.random.seed(0)
torch.manual_seed(0)

# 1. Define a simple policy network for the CartPole environment

class Policy(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

policy_model = Policy(input_dim=4, output_dim=2)

# 2. Training Algorithm

def policy_gradient_loss(logits: torch.Tensor,
                         actions: torch.Tensor,
                         advantages: torch.Tensor
                         ) -> torch.Tensor:
    """
    logits: [batch, num_actions]
    actions: [batch] (int64)
    advantages: [batch] (float)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    chosen_log_probs = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    loss = -(chosen_log_probs * advantages).mean()
    return loss

num_episodes = 400
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-3)
gamma = 0.99

# 3. Environment setup

env = gym.make("CartPole-v1")
observations, info = env.reset()

def discounted_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


# 4. Training loop
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

env.close()

# 5. Plotting results

fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, "o-")
plt.title("Policy Loss")
plt.xlabel("Environments")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(completed_rewards, "o-", label="Episode return")
window = min(20, len(completed_rewards))
if window >= 2:
    rolling = np.convolve(completed_rewards, np.ones(window) / window, mode="valid")
    rolling_steps = list(range(window - 1, len(completed_rewards)))
    plt.plot(rolling_steps, rolling, "-", label=f"{window}-ep mean")
plt.title("Episode Returns")
plt.xlabel("Environments")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
plt.show()

figures_dir = "experiments/cartpole/figures/"
os.makedirs(figures_dir, exist_ok=True)
fig.savefig(f"{figures_dir}cartpole_training.png")