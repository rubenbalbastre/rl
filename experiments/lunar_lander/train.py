import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import random


# Running mean/std for observation normalization
class RunningMeanStd:
    def __init__(self, shape, eps=1e-8):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = eps

    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)
    

# 1. Environment setup
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
n_envs = 8
envs = gym.vector.SyncVectorEnv([lambda: gym.make(
        "LunarLander-v3", 
        continuous=True, 
        enable_wind=False
    ) for _ in range(n_envs)
], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
envs.action_space.seed(seed)
observations, info = envs.reset(seed=seed)
obs_rms = RunningMeanStd(shape=observations.shape[-1:])

action_space_low = torch.tensor(envs.action_space.low, dtype=torch.float32)
action_space_high = torch.tensor(envs.action_space.high, dtype=torch.float32)


# 2. Define a simple policy and value network for the LunarLander environment

class PolicyValueModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyValueModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = torch.nn.Linear(hidden_dim, output_dim)  # For mean of action distribution
        self.value_head = torch.nn.Linear(hidden_dim, 1)  # For value estimation
        self.log_std = torch.nn.Parameter(torch.zeros(output_dim))  # Learnable log standard deviation for continuous actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.policy_head(x)
        value = self.value_head(x)
        return mu, self.log_std, value
    

action_dim_space = envs.action_space.shape[-1]
observation_dim_space = envs.observation_space.shape[-1]
print(f"Action space dimension: {action_dim_space}, Observation space dimension: {observation_dim_space}")
policy_value_model = PolicyValueModel(input_dim=observation_dim_space, output_dim=action_dim_space)


# 3. Training Algorithm

optimizer = torch.optim.Adam(
    list(policy_value_model.parameters()), 
    lr=3e-4
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
    """
    rewards, values, dones: shape (T, n_envs)
    next_value: shape (n_envs,) or scalar
    """
    T, n_envs = rewards.shape
    adv = torch.zeros_like(rewards, dtype=torch.float32)
    last_gae = torch.zeros((n_envs,), dtype=torch.float32)

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae

    returns = adv + values
    return adv, returns


# 4. Training loop

num_updates = 1000
num_rollouts_per_env = 256
num_rollouts = num_rollouts_per_env * n_envs
batch_size = num_rollouts_per_env
ppo_epochs = 4
clip_epsilon = 0.2
num_optimizer_steps = (num_rollouts + batch_size - 1) // batch_size


# Metrics
episode_return_rollout = torch.zeros(n_envs, dtype=torch.float32)
episode_returns = []
policy_losses = []
value_losses = []
entropies = []
approx_kls = []
clip_fractions = []

for update in tqdm(range(num_updates), desc="Training PPO"):

    # Collect trajectories
    observations_rollout = torch.empty((num_rollouts_per_env, n_envs, observation_dim_space))
    log_probs_rollout = torch.empty((num_rollouts_per_env, n_envs))
    rewards_rollout = torch.empty((num_rollouts_per_env, n_envs))
    values_rollout = torch.empty((num_rollouts_per_env, n_envs))
    action_rollout = torch.empty((num_rollouts_per_env, n_envs, action_dim_space))
    dones_rollout = torch.empty((num_rollouts_per_env, n_envs))

    with torch.no_grad():

        for rollout_id in range(num_rollouts_per_env):

            # 1. get actions from the policy and value estimates for the current observations
            obs_rms.update(observations)
            obs_tensor = obs_rms.normalize(observations)
            mu, log_std, value = policy_value_model(obs_tensor)
            std = torch.exp(log_std)

            action_dist = torch.distributions.Normal(mu, std)
            pre_tanh_action = action_dist.rsample()
            tanh_action = torch.tanh(pre_tanh_action)
            action_scaled = action_space_low + (tanh_action + 1.0) * 0.5 * (action_space_high - action_space_low)

            # Tanh-squash correction
            log_prob = action_dist.log_prob(pre_tanh_action).sum(-1)
            log_prob -= torch.log(1.0 - tanh_action.pow(2) + 1e-6).sum(-1)

            # 2. get next observations and rewards from the environment
            observations, reward, done, truncated, info = envs.step(action_scaled.numpy())
            termination_flag = torch.from_numpy(done | truncated)

            # 3. logging
            values_rollout[rollout_id] = value.squeeze(-1)
            observations_rollout[rollout_id] = obs_tensor
            log_probs_rollout[rollout_id] = log_prob
            rewards_rollout[rollout_id] = torch.from_numpy(reward)
            action_rollout[rollout_id] = pre_tanh_action
            dones_rollout[rollout_id] = termination_flag
            episode_return_rollout += torch.from_numpy(reward)
            
            if termination_flag.any():
                terminated_ids = torch.where(termination_flag)[0]
                episode_returns.append(episode_return_rollout[terminated_ids].mean().item())
                for idx in terminated_ids.tolist():
                    episode_return_rollout[idx] = 0.0
                

    # Compute advantages
    with torch.no_grad():
        # Always compute next_value for each env; done handling is managed by the mask in GAE.
        next_obs_tensor = obs_rms.normalize(observations)
        _, _, next_value = policy_value_model(next_obs_tensor)
        next_value = next_value.squeeze(-1)

    advantages, returns = compute_gae(
        rewards_rollout,
        values=values_rollout,
        dones=dones_rollout,
        next_value=next_value,
        gamma=gamma,
    )
    # Reshape rollouts to (num_rollouts*n_envs, ...) for easier batching
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    rewards_rollout = rewards_rollout.view(num_rollouts)
    values_rollout = values_rollout.view(num_rollouts)
    dones_rollout = dones_rollout.view(num_rollouts)
    advantages = advantages.view(num_rollouts)
    old_log_probs = log_probs_rollout.view(-1)
    action_rollout = action_rollout.view(-1, action_dim_space)
    observations_rollout = observations_rollout.view(-1, observation_dim_space)
    returns = returns.view(num_rollouts)
    
    # PPO update
    for epoch in range(ppo_epochs):
        inds = torch.randperm(num_rollouts)
        for id in range(0, num_rollouts, batch_size):

            # get batches of data
            mb_inds = inds[id : id + batch_size]
            obs_batch = observations_rollout[mb_inds]
            action_batch = action_rollout[mb_inds]
            old_log_probs_batch = old_log_probs[mb_inds]
            advantages_batch = advantages[mb_inds]
            returns_batch = returns[mb_inds]

            # get policy outputs and values for the current batch of observations
            logits, log_std, values = policy_value_model(obs_batch)
            std = torch.exp(log_std)
            action_dist = torch.distributions.Normal(logits, std)
            new_log_probs = action_dist.log_prob(action_batch).sum(-1)
            tanh_action = torch.tanh(action_batch)
            new_log_probs -= torch.log(1.0 - tanh_action.pow(2) + 1e-6).sum(-1)

            # Loss computation
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * advantages_batch, clipped_ratio * advantages_batch).mean()
            value_loss = 0.5 * (returns_batch - values.squeeze(-1)).pow(2).mean()
            entropy_bonus = action_dist.entropy().mean()
            loss = compute_ppo_loss(
                policy_loss, 
                value_loss, 
                entropy_bonus, 
                value_loss_coef=0.5, 
                entropy_coef=0.01
            )

            # Logging metrics
            approx_kl = (old_log_probs_batch - new_log_probs).mean()
            clip_frac = ((ratio > (1 + clip_epsilon)) | (ratio < (1 - clip_epsilon))).float().mean()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_bonus.item())
            approx_kls.append(approx_kl.item())
            clip_fractions.append(clip_frac.item())

            # Update policy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy_value_model.parameters()), max_norm=0.5)  # Gradient clipping
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
