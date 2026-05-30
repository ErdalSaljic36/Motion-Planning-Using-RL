"""Quick sanity check that MazeEnv works."""

from maze_env import MazeEnv

env = MazeEnv()
obs, info = env.reset()
print(f"Initial observation: {obs}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print(f"Maze shape: {env.maze.shape}")

# Try a few steps
print("\n--- Taking 5 random actions ---")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: action={action}, obs={obs}, reward={reward:.2f}, "
          f"terminated={terminated}, truncated={truncated}")

env.close()
print("\n✓ Environment works!")