import time
from maze_env import MazeEnv

env = MazeEnv(render_mode="human")
obs, info = env.reset()
print(f"Initial observation: {obs}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print(f"Maze shape: {env.maze.shape}")

env.render()
time.sleep(1)


print("Agent navigating in the maze")
total_reward=0

#Loop
for step in range(200):
    action=env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    env.render()


    if terminated:
        print(f"\n✓ Reached the goal in {step + 1} steps! Total reward: {total_reward:.2f}")
        time.sleep(2)  # pause to see win
        break
    if truncated:
        print(f"\n✗ Episode truncated after {step + 1} steps. Total reward: {total_reward:.2f}")
        time.sleep(2)
        break


env.close()
print("\n✓ Environment works!")