import torch
import gymnasium as gym
import matplotlib.pyplot as plt


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

optimizer = torch.optim.Adam(list(policy_model.parameters()) + list(value_model.parameters()), lr=1e-3)
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

num_updates = 10
batch_size = 64
ppo_epochs = 4

for update in range(num_updates):

    # Collect trajectories
    observations_rollout = []
    log_probs_rollout = []
    rewards_rollout = []
    values_rollout = []
    action_rollout = []

    with torch.no_grad():
        for _ in range(batch_size):

            # 1. get actions from the policy
            obs_tensor = torch.tensor(observations, dtype=torch.float32)
            mu, log_std = policy_model(obs_tensor)
            std = torch.exp(log_std)

            action_dist = torch.distributions.Normal(mu, std)
            action = action_dist.sample()
            action = torch.clamp(action, torch.tensor(env.action_space.low), torch.tensor(env.action_space.high))  # Ensure actions are within bounds

            log_prob = action_dist.log_prob(action).sum(-1)  # Sum log probs for multi-dimensional actions

            # 2. get value estimates for the current observations
            value = value_model(obs_tensor)
            values_rollout.append(value)

            # 3. get next observations and rewards from the environment
            observations, reward, done, truncated, info = env.step(action.numpy())
            
            # 4. logging
            observations_rollout.append(obs_tensor)
            log_probs_rollout.append(log_prob)
            rewards_rollout.append(reward)
            action_rollout.append(action)

            if done or truncated:
                observations, info = env.reset()

    # Compute advantages
    advantages, returns = compute_gae(rewards_rollout, values=torch.stack(values_rollout), dones=[0]*len(rewards_rollout), next_value=0, gamma=gamma)
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
        policy_loss = compute_clipped_ppo_policy_loss(old_log_probs, new_log_probs, advantages)
        value_loss = 0.5 * (returns - values.squeeze(-1)).pow(2).mean()
        entropy_bonus = action_dist.entropy().mean()
        loss = compute_ppo_loss(policy_loss, value_loss, entropy_bonus, value_loss_coef=0.5, entropy_coef=0.01)

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy_model.parameters()) + list(value_model.parameters()), max_norm=0.5)  # Gradient clipping
        optimizer.step()

        print(f"Update {update+1}/{num_updates}, Epoch {epoch+1}/{ppo_epochs}, Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy Bonus: {entropy_bonus.item():.4f}")
