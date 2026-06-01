#Statistical analysis of random agent after 1000 episodes ; used as a baseline for comparison with later algorithms

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from maze_env import MazeEnv

NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200
SEED = 42 

OUTPUT_DIR = os.path.join("..", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Set NumPy seed for reproducibility (agent's randomness)
np.random.seed(SEED)

env = MazeEnv(max_steps=MAX_STEPS_PER_EPISODE)

# Storage for per-episode statistics
episode_rewards = []
episode_lengths = []
episode_successes = []  # True if reached goal, False otherwise
episode_wall_hits = []  # number of times agent hit a wall

# Visitation heatmap: counts how many times each cell is visited (across all episodes)
visitation_map = np.zeros((env.maze_size, env.maze_size), dtype=np.int64)

#Loop for episodes
for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed=SEED + episode)  # different seed per episode
    
    total_reward = 0.0
    steps = 0
    wall_hits = 0
    
    #Loop for one episode
    for step in range(MAX_STEPS_PER_EPISODE):
        # Track visitation
        row, col = obs[0], obs[1]
        visitation_map[row, col] += 1
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Count wall hits 
        if reward == -0.1:
            wall_hits += 1
        
        if terminated or truncated:
            break
    
    
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    episode_successes.append(terminated)  
    episode_wall_hits.append(wall_hits)

env.close()
print("-" * 60)

episode_rewards = np.array(episode_rewards)
episode_lengths = np.array(episode_lengths)
episode_successes = np.array(episode_successes)
episode_wall_hits = np.array(episode_wall_hits)

success_rate = episode_successes.mean() * 100
mean_reward = episode_rewards.mean()
std_reward = episode_rewards.std()
mean_length = episode_lengths.mean()
mean_wall_hits = episode_wall_hits.mean()

# Stats for successful episodes only
successful_mask = episode_successes
if successful_mask.any():
    mean_length_successful = episode_lengths[successful_mask].mean()
    mean_reward_successful = episode_rewards[successful_mask].mean()
else:
    mean_length_successful = None
    mean_reward_successful = None

print("\n" + "=" * 60)
print("RANDOM AGENT BASELINE — RESULTS")
print(f"Total episodes:           {NUM_EPISODES}")
print(f"Successful episodes:      {int(episode_successes.sum())} "
      f"({success_rate:.2f}%)")
print(f"Mean total reward:        {mean_reward:.3f} (± {std_reward:.3f})")
print(f"Mean episode length:      {mean_length:.1f} steps")
print(f"Mean wall hits/episode:   {mean_wall_hits:.1f}")

if mean_length_successful is not None:
    print(f"\nFor successful episodes only:")
    print(f"  Mean length:            {mean_length_successful:.1f} steps")
    print(f"  Mean reward:            {mean_reward_successful:.3f}")
print("=" * 60)


csv_path = os.path.join(OUTPUT_DIR, "random_agent_episodes.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "total_reward", "length", "success", "wall_hits"])
    for i in range(NUM_EPISODES):
        writer.writerow([
            i,
            episode_rewards[i],
            episode_lengths[i],
            int(episode_successes[i]),
            episode_wall_hits[i],
        ])
print(f"\nRaw episode data saved to: {csv_path}")


# Figure 1: Reward distribution histogram

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(episode_rewards, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
ax.axvline(mean_reward, color="red", linestyle="--", linewidth=2,
           label=f"Mean = {mean_reward:.2f}")
ax.set_xlabel("Total reward per episode")
ax.set_ylabel("Number of episodes")
ax.set_title(f"Random Agent: Reward Distribution ({NUM_EPISODES} episodes)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "random_agent_reward_histogram.png")
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"Figure saved: {fig_path}")



# Figure 2: Episode length distribution

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(episode_lengths, bins=50, color="seagreen", edgecolor="black", alpha=0.7)
ax.axvline(mean_length, color="red", linestyle="--", linewidth=2,
           label=f"Mean = {mean_length:.1f}")
ax.set_xlabel("Episode length (steps)")
ax.set_ylabel("Number of episodes")
ax.set_title(f"Random Agent: Episode Length Distribution ({NUM_EPISODES} episodes)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "random_agent_length_histogram.png")
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"Figure saved: {fig_path}")



# Figure 3: Cumulative success rate over episodes

cumulative_successes = np.cumsum(episode_successes)
cumulative_success_rate = cumulative_successes / np.arange(1, NUM_EPISODES + 1) * 100

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cumulative_success_rate, color="darkorange", linewidth=2)
ax.axhline(success_rate, color="red", linestyle="--",
           label=f"Final success rate = {success_rate:.2f}%")
ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative success rate (%)")
ax.set_title(f"Random Agent: Cumulative Success Rate over {NUM_EPISODES} Episodes")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "random_agent_success_rate.png")
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"Figure saved: {fig_path}")




# Figure 4: Visitation heatmap

fig, ax = plt.subplots(figsize=(8, 8))

# Mask walls (don't show heat in wall cells, they should be black)
masked_visitation = np.ma.masked_where(env.maze == 1, visitation_map)

cmap = plt.cm.hot
cmap.set_bad(color="black")

im = ax.imshow(masked_visitation, cmap=cmap, interpolation="nearest")

# Mark start and goal
ax.scatter(env.start_pos[1], env.start_pos[0], marker="s", s=200,
           edgecolor="cyan", facecolor="none", linewidth=3, label="Start")
ax.scatter(env.goal_pos[1], env.goal_pos[0], marker="*", s=300,
           edgecolor="lime", facecolor="lime", label="Goal")

ax.set_title(f"Random Agent: Visitation Heatmap ({NUM_EPISODES} episodes)")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.legend(loc="upper right")
plt.colorbar(im, ax=ax, label="Number of visits", fraction=0.046, pad=0.04)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "random_agent_visitation_heatmap.png")
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"Figure saved: {fig_path}")


print("\n✓ All analyses complete!")