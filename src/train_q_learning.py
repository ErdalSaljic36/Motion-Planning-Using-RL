#Training the agent using tabular Q-Learning on MazeEnv

import numpy as np
import matplotlib.pyplot as plt
import os
from maze_env import MazeEnv
from q_learning_agent import QLearningAgent

NUM_EPISODES = 5000
MAX_STEP_EPISODE = 200
SEED = 42

Alpha = 0.01
Gamma = 0.99
Eps_start = 1.0
Eps_min = 0.05
Eps_decay = 0.995

FIGURES_DIR = os.path.join("..", "figures")
MODELS_DIR = os.path.join("..", "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

env = MazeEnv(max_steps=MAX_STEP_EPISODE)
agent = QLearningAgent(maze_size= env.maze_size, num_actions=env.action_space.n, alpha= Alpha, gamma= Gamma, eps_start= Eps_start, eps_min= Eps_min, eps_decay= Eps_decay, seed= SEED)


#Training loop

episode_rewards=[]
episode_successes=[]
episode_lengths =[]
epsilon_history=[]


for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed = SEED + episode)
    total_reward = 0.0
    steps = 0

    for step in range(MAX_STEP_EPISODE):
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated)

        #Move to the next step
        obs = next_obs
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break
    agent.decay_eps()

    episode_rewards.append(reward)
    episode_successes.append(terminated)
    episode_lengths.append(steps)
    epsilon_history.append(agent.eps)

    if (episode + 1) % 500 == 0:
        # Average over the last 500 episodes
        recent_rewards = episode_rewards[-500:]
        recent_successes = episode_successes[-500:]
        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_successes) * 100
        print(f"  Episode {episode + 1}/{NUM_EPISODES} | "
              f"avg reward (last 500): {avg_reward:6.3f} | "
              f"success rate: {success_rate:5.1f}% | "
              f"epsilon: {agent.eps:.3f}")

env.close()
episode_rewards = np.array(episode_rewards)
episode_lengths = np.array(episode_lengths)
episode_successes = np.array(episode_successes)
epsilon_history = np.array(epsilon_history)

final_rewards = episode_rewards[-500:]
final_lengths = episode_lengths[-500:]
final_successes = episode_successes[-500:]

print("=" * 60)
print("Q-LEARNING TRAINING — RESULTS (last 500 episodes)")
print(f"Success rate:        {final_successes.mean() * 100:.2f}%")
print(f"Mean reward:         {final_rewards.mean():.3f} "
      f"(± {final_rewards.std():.3f})")
print(f"Mean episode length: {final_lengths.mean():.1f} steps")
print("=" * 60)
print("\nComparison to random agent baseline:")
print(f"  Random success rate:  ~7.9%   -> Q-Learning: {final_successes.mean()*100:.1f}%")
print(f"  Random mean reward:   ~-10.4  -> Q-Learning: {final_rewards.mean():.2f}")
print(f"  Random mean length:   ~196    -> Q-Learning: {final_lengths.mean():.0f}")


q_table_path = os.path.join(MODELS_DIR, "q_table.npy")
np.save(q_table_path, agent.q_table)
print(f"\nLearned Q-table saved to: {q_table_path}")

#Helper funtion to smooth out the data
def moving_average(data, window=100):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")



#Reward per episode
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(episode_rewards, alpha=0.3, color="steelblue", label="Reward per episode")
smoothed = moving_average(episode_rewards, window=100)
ax.plot(range(len(smoothed)), smoothed, color="darkblue", linewidth=2,
        label="Moving average (100 episodes)")
ax.set_xlabel("Episode")
ax.set_ylabel("Total reward")
ax.set_title("Q-Learning: Reward per Episode")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "qlearning_reward_curve.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")


#Episode length
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(episode_lengths, alpha=0.3, color="seagreen", label="Length per episode")
smoothed = moving_average(episode_lengths, window=100)
ax.plot(range(len(smoothed)), smoothed, color="darkgreen", linewidth=2,
        label="Moving average (100 episodes)")
ax.set_xlabel("Episode")
ax.set_ylabel("Episode length (steps)")
ax.set_title("Q-Learning: Episode Length per Episode")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "qlearning_length_curve.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")

#Success rate
success_rolling = moving_average(episode_successes.astype(float), window=100) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(success_rolling)), success_rolling, color="darkorange",
        linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Success rate (%, rolling 100 episodes)")
ax.set_title("Q-Learning: Success Rate over Training")
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "qlearning_success_curve.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")

#Epsilon decay
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epsilon_history, color="purple", linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Epsilon (exploration rate)")
ax.set_title("Q-Learning: Epsilon Decay over Training")
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "qlearning_epsilon_decay.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")


print("\n✓ Training complete!")

