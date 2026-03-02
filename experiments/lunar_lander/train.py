import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# 1. Environment setup

env = gym.make("LunarLander-v3", continuous=True, enable_wind=False)
observations, info = env.reset()


# 2. Define a simple policy network for the LunarLander environment

class Policy(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)

        self.log_std = torch.nn.Parameter(torch.zeros(output_dim))  # Learnable log standard deviation for continuous actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.fc2(x)
        return mu, self.log_std
    

class ValueFunction(torch.nn.Module):
    def __init__(self, input_dim):
        super(ValueFunction, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value.squeeze(-1)  # Return a scalar value


action_dim_space = env.action_space.shape[0]
observation_dim_space = env.observation_space.shape[0]
print(f"Action space dimension: {action_dim_space}, Observation space dimension: {observation_dim_space}")
policy_model = Policy(input_dim=observation_dim_space, output_dim=action_dim_space)
value_model = ValueFunction(input_dim=observation_dim_space)


# 3. Training Algorithm

optimizer = torch.optim.Adam(
    list(policy_model.parameters()) + list(value_model.parameters()), 
    lr=1e-3
)
gamma = 0.99


def compute_ppo_loss(policy_loss, value_loss, entropy_bonus, value_loss_coef=0.5, entropy_coef=0.01):
    total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus
    return total_loss


def compute_clipped_ppo_policy_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T)
    last_gae = 0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * (next_value if t==T-1 else values[t+1]) * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


# 4. Training loop

num_updates = 3000
batch_size = 512
ppo_epochs = 8
clip_epsilon = 0.2

# Metrics
episode_returns = []
episode_return = 0.0
policy_losses = []
value_losses = []
entropies = []
approx_kls = []
clip_fractions = []

for update in tqdm(range(num_updates), desc="Training PPO"):

    # Collect trajectories
    observations_rollout = []
    log_probs_rollout = []
    rewards_rollout = []
    values_rollout = []
    action_rollout = []
    dones_rollout = []

    with torch.no_grad():
        for _ in range(batch_size):

            # 1. get actions from the policy
            obs_tensor = torch.tensor(observations, dtype=torch.float32)
            mu, log_std = policy_model(obs_tensor)
            std = torch.exp(log_std)

            action_dist = torch.distributions.Normal(mu, std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(-1)  # Sum log probs for multi-dimensional actions
            action_clipped = torch.clamp(
                action,
                torch.tensor(env.action_space.low),
                torch.tensor(env.action_space.high),
            )

            # 2. get value estimates for the current observations
            value = value_model(obs_tensor)
            values_rollout.append(value)

            # 3. get next observations and rewards from the environment
            observations, reward, done, truncated, info = env.step(action_clipped.numpy())
            
            # 4. logging
            observations_rollout.append(obs_tensor)
            log_probs_rollout.append(log_prob)
            rewards_rollout.append(reward)
            action_rollout.append(action)
            dones_rollout.append(done or truncated)
            episode_return += reward

            if done or truncated:
                episode_returns.append(episode_return)
                episode_return = 0.0
                observations, info = env.reset()

    # Compute advantages
    with torch.no_grad():
        if len(dones_rollout) > 0 and not dones_rollout[-1]:
            next_value = value_model(torch.tensor(observations, dtype=torch.float32)).item()
        else:
            next_value = 0.0

    advantages, returns = compute_gae(
        rewards_rollout,
        values=torch.stack(values_rollout),
        dones=dones_rollout,
        next_value=next_value,
        gamma=gamma,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    old_log_probs = torch.stack(log_probs_rollout)
    action_rollout = torch.stack(action_rollout)
    observations_rollout = torch.stack(observations_rollout)

    for epoch in range(ppo_epochs):

        # get policy outputs for the current batch of observations
        logits, log_std = policy_model(observations_rollout)
        std = torch.exp(log_std)
        action_dist = torch.distributions.Normal(logits, std)
        new_log_probs = action_dist.log_prob(action_rollout).sum(-1)  # Sum log probs for multi-dimensional actions

        # get values for the current batch of observations
        values = value_model(observations_rollout)

        # Loss computation
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = 0.5 * (returns - values.squeeze(-1)).pow(2).mean()
        entropy_bonus = action_dist.entropy().mean()
        loss = compute_ppo_loss(policy_loss, value_loss, entropy_bonus, value_loss_coef=0.5, entropy_coef=0.0001)

        # Metrics
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_frac = ((ratio > (1 + clip_epsilon)) | (ratio < (1 - clip_epsilon))).float().mean()
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropies.append(entropy_bonus.item())
        approx_kls.append(approx_kl.item())
        clip_fractions.append(clip_frac.item())

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy_model.parameters()) + list(value_model.parameters()), max_norm=0.5)  # Gradient clipping
        optimizer.step()

    # Early stopping if solved
    if len(episode_returns) >= 100:
        avg100 = sum(episode_returns[-100:]) / 100
        if avg100 >= 200:
            print(f"Solved at update {update + 1}, avg100={avg100:.1f}")
            break

# 5. Plotting results

fig, axes = plt.subplots(3, 2, figsize=(12, 8), dpi=120)

def moving_average(values, window):
    if len(values) < window:
        return []
    return (
        torch.tensor(values, dtype=torch.float32)
        .unfold(0, window, 1)
        .mean(dim=1)
        .numpy()
    )

# Episode return
ax = axes[0, 0]
ax.plot(episode_returns, color="#1f77b4", alpha=0.35, label="Episode return")
if len(episode_returns) >= 10:
    window = min(50, len(episode_returns))
    rolling = moving_average(episode_returns, window)
    ax.plot(range(window - 1, len(episode_returns)), rolling, color="#1f77b4", label=f"{window}-ep mean")
ax.set_title("Episode Return")
ax.set_xlabel("Updates")
ax.set_ylabel("Return")
ax.legend(frameon=False)
ax.grid(True)

# Policy
ax = axes[0, 1]
ax.set_title("Policy loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Policy loss")
line1 = ax.plot(policy_losses, color="#1f77b4", alpha=0.35, label="Policy loss")
ax.grid(True)
if len(policy_losses) >= 10:
    window = min(50, len(policy_losses))
    rolling = moving_average(policy_losses, window)
    ax.plot(range(window - 1, len(policy_losses)), rolling, color="#1f77b4", label=f"{window}-ep mean")

# Value loss
ax2 = axes[1, 1]
ax2.set_ylabel("Value loss")
ax2.set_title("Value loss")
ax2.set_xlabel("Epochs")
line2 = ax2.plot(value_losses, color="#ff7f0e", alpha=0.35, label="Value loss")
ax2.grid(True)
if len(value_losses) >= 10:
    window = min(50, len(value_losses))
    rolling = moving_average(value_losses, window)
    ax2.plot(range(window - 1, len(value_losses)), rolling, color="#ff7f0e", label=f"{window}-ep mean")

# Entropy
ax = axes[1, 0]
ax.plot(entropies, color="#2ca02c", alpha=0.35, label="Entropy")
ax.set_title("Entropy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Entropy")
ax.grid(True)
if len(entropies) >= 10:
    window = min(50, len(entropies))
    rolling = moving_average(entropies, window)
    ax.plot(range(window - 1, len(entropies)), rolling, color="#2ca02c", label=f"{window}-ep mean")

# Approx KL / Clip fraction (separate y-axes)
ax = axes[2, 1]
ax.set_title("Approx KL")
ax.set_xlabel("Epochs")
ax.set_ylabel("Approx KL")
ax.plot(approx_kls, color="#1f77b4", alpha=0.35, label="Approx KL")
ax.grid(True)
if len(approx_kls) >= 10:
    window = min(50, len(approx_kls))
    rolling = moving_average(approx_kls, window)
    ax.plot(range(window - 1, len(approx_kls)), rolling, color="#1f77b4", label=f"{window}-ep mean")

ax = axes[2, 0]
ax.set_title("Clip fraction")
ax.set_xlabel("Epochs")
ax.set_ylabel("Clip fraction")
ax.plot(clip_fractions, color="#ff7f0e", alpha=0.35, label="Clip fraction")
ax.grid(True)
if len(clip_fractions) >= 10:
    window = min(50, len(clip_fractions))
    rolling = moving_average(clip_fractions, window)
    ax.plot(range(window - 1, len(clip_fractions)), rolling, color="#ff7f0e", label=f"{window}-ep mean")

fig.tight_layout()

figures_dir = "experiments/lunar_lander/figures/"
os.makedirs(figures_dir, exist_ok=True)
fig.savefig(f"{figures_dir}lunar_lander_training.png")
