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

num_steps = 10000
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-3)
batch_size = 8
gamma = 0.99

# 3. Environment setup

envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(batch_size)],
    autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
)
observations, info = envs.reset()

def discounted_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


# logging variables
losses = []
loss_steps = []
episode_observations = [[] for _ in range(batch_size)]
episode_actions = [[] for _ in range(batch_size)]
episode_rewards = [[] for _ in range(batch_size)]
completed_rewards = []
completed_steps = []
total_env_steps = 0

# 4. Training loop

for epoch_id in range(num_steps):

    # get action from policy
    torch_observations = torch.tensor(observations, dtype=torch.float32)
    logits = policy_model(torch_observations)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()

    # get observation from environment
    observations, rewards, terminated, truncated, info = envs.step(actions.numpy())

    # logging
    total_env_steps += batch_size
    for i in range(batch_size):
        episode_observations[i].append(observations[i])
        episode_actions[i].append(actions[i].item())
        episode_rewards[i].append(rewards[i])

    # update only for completed episodes
    done = terminated | truncated
    if done.any():

        # zero gradients
        optimizer.zero_grad()

        # compute loss for completed episodes and update policy
        done_indices = np.where(done)[0].tolist()
        per_episode_losses = []
        for i in done_indices:

            obs_tensor = torch.tensor(
                np.array(episode_observations[i], dtype=np.float32),
                dtype=torch.float32,
            )
            act_tensor = torch.tensor(episode_actions[i], dtype=torch.int64)
            
            # compute policy
            logits_ep = policy_model(obs_tensor)
            log_probs_ep = F.log_softmax(logits_ep, dim=-1)

            # compute discounted returns and advantages
            returns = discounted_returns(episode_rewards[i], gamma=gamma)
            advantages = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
            
            # compute loss for the episode
            loss = policy_gradient_loss(logits_ep, act_tensor, advantages)
            per_episode_losses.append(loss)

            # log episode return and reset episode data
            completed_rewards.append(float(sum(episode_rewards[i])))
            completed_steps.append(total_env_steps)
            episode_observations[i] = []
            episode_actions[i] = []
            episode_rewards[i] = []

        # compute mean loss across completed episodes and update policy
        loss = torch.stack(per_episode_losses).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loss_steps.append(total_env_steps)

envs.close()

# 5. Plotting results

fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
if losses:
    plt.plot(loss_steps, losses, "o-")
plt.title("Policy Loss")
plt.xlabel("Environment Steps")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
if completed_rewards:
    plt.plot(completed_steps, completed_rewards, "o-", label="Episode return")
    window = min(20, len(completed_rewards))
    if window >= 2:
        rolling = np.convolve(completed_rewards, np.ones(window) / window, mode="valid")
        rolling_steps = completed_steps[window - 1:]
        plt.plot(rolling_steps, rolling, "-", label=f"{window}-ep mean")
plt.title("Episode Returns")
plt.xlabel("Environment Steps")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
plt.show()

figures_dir = "experiments/cartpole/figures/"
os.makedirs(figures_dir, exist_ok=True)
fig.savefig(f"{figures_dir}cartpole_training.png")