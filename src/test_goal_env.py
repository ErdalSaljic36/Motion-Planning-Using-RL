import pickle
import numpy as np
from maze_env_goal import MazeEnvGoal

with open("../datasets/train_mazes.pkl", "rb") as f:
    train_mazes = pickle.load(f)

# Create goal-conditioned env with 3x3 window
env = MazeEnvGoal(window_size=3, maze_set=train_mazes)

print("Observation space:", env.observation_space)
print(f"Observation dim: {env.observation_space.shape[0]} "
      f"(expected {3*3} window + 2 goal = 11)\n")

obs, info = env.reset()
print(f"Start: {env.start_pos}, Goal: {env.goal_pos}, maze_size: {env.maze_size}")
print(f"Observation ({len(obs)} values):")
print(f"  Window part (first 9):  {obs[:9]}")
print(f"  Goal part (last 2):     {obs[9:]}")

# Verify goal signal makes sense
expected_drow = (env.goal_pos[0] - env.agent_pos[0]) / env.maze_size
expected_dcol = (env.goal_pos[1] - env.agent_pos[1]) / env.maze_size
print(f"\nExpected goal signal: [{expected_drow:.3f}, {expected_dcol:.3f}]")
print(f"Actual goal signal:   {obs[9:]}")

# Take a step and see goal signal change
obs2, reward, term, trunc, info = env.step(env.action_space.sample())
print(f"\nAfter one step, goal signal: {obs2[9:]} (should change if agent moved)")

print("\n✓ MazeEnvGoal works!")