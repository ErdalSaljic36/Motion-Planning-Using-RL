#Used to visualize the learned policy and movement of the robot
import time
import numpy as np
from maze_env import MazeEnv
from q_learning_agent import QLearningAgent
import os


MODELS_DIR = os.path.join("..", "models")
Q_TABLE_PATH = os.path.join(MODELS_DIR, "q_table.npy")

MAX_STEPS = 200
STEP_DELAY = 0.2  # seconds between steps



env = MazeEnv(max_steps=MAX_STEPS, render_mode="human")

agent = QLearningAgent(
    maze_size=env.maze_size,
    num_actions=env.action_space.n,
)

# Load the learned Q-table
agent.q_table = np.load(Q_TABLE_PATH)

# Pure exploitation — always pick the best action
agent.eps = 0.0

print("Loaded trained Q-table.")
print(f"Q-table shape: {agent.q_table.shape}")
print("Running one episode with the trained agent...")
print("-" * 50)

#One episode with visualization
obs, info = env.reset(seed=42)
env.render()
time.sleep(1)  # pause to see the start

total_reward = 0.0
path = [tuple(obs)]  # track the agent's path

for step in range(MAX_STEPS):
    action = agent.choose_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    path.append(tuple(obs))

    env.render()
    time.sleep(STEP_DELAY)

    if terminated:
        print(f"  Reached the goal in {step + 1} steps!")
        print(f"  Total reward: {total_reward:.3f}")
        time.sleep(2)  
        break
    if truncated:
        print(f"  Episode truncated after {step + 1} steps.")
        print(f"  Total reward: {total_reward:.3f}")
        break

env.close()


print(f"Path taken ({len(path)} positions):")
print(path)
print("\n Evaluation complete!")