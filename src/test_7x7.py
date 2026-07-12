"""Inspect what the 7x7 observation actually looks like at different positions."""

import pickle
import numpy as np
from maze_env_goal import MazeEnvGoal

with open("../datasets/train_mazes.pkl", "rb") as f:
    train_mazes = pickle.load(f)

env = MazeEnvGoal(window_size=7, maze_set=[train_mazes[0]])
obs, info = env.reset()

print(f"Maze size: {env.maze_size}")
print(f"Agent start: {env.agent_pos}, Goal: {env.goal_pos}")
print(f"Observation shape: {obs.shape}")
print(f"\nFull maze:")
print(env.maze)
print(f"\nWindow part (7x7 = 49 values), reshaped:")
window_part = obs[:49].reshape(7, 7)
print(window_part)
print(f"\nGoal signal (last 2): {obs[49:]}")

# Count how many cells are "wall/out-of-bounds" (1) vs empty (0)
walls = np.sum(window_part == 1)
print(f"\nWall/OOB cells in window: {walls}/49 ({walls/49*100:.0f}%)")