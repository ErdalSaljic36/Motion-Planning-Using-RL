import pickle
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from maze_env_goal import MazeEnvGoal


WINDOW_SIZE=3
TOTAL_TIMESTEPS = 100000
SEED = 42

with open("../datasets/train_mazes.pkl", "rb") as f:
    train_mazes = pickle.load(f)

single_maze = [train_mazes[0]]   # a list with just one maze
print(f"Training on a single maze:")
print(f"  start={train_mazes[0]['start']}, goal={train_mazes[0]['goal']}, "
      f"optimal_len={train_mazes[0]['optimal_len']}")


env = MazeEnvGoal(window_size=WINDOW_SIZE, maze_set=single_maze)
env = Monitor(env)


model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50000,learning_starts=1000, batch_size=64, gamma=0.99, target_update_interval=1000, exploration_fraction=0.3, exploration_final_eps=0.05, verbose=1, seed=SEED)

print(f"\nTraining DQN for {TOTAL_TIMESTEPS:,} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

model.save("../models/dqn_single_maze")
print("\nModel saved to ../models/dqn_single_maze")

print("\n--- Evaluation (10 episodes) ---")


eval_env = MazeEnvGoal(window_size=WINDOW_SIZE, maze_set=single_maze)

successes = 0
lengths = []
for ep in range(10):
    obs, info = eval_env.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(int(action))
        steps += 1
    if terminated:  # reached goal
        successes += 1
        lengths.append(steps)
    print(f"Episode {ep+1}: {'SUCCESS' if terminated else 'FAIL'} in {steps} steps")

print(f"\nSuccess rate: {successes}/10")
if lengths:
    print(f"Average steps when successful: {np.mean(lengths):.1f} "
          f"(optimal: {train_mazes[0]['optimal_len']})")

print("\n✓ Single-maze DQN training complete!")