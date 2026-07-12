#DQN training for GENERALIZATION across mazes.
import os
import csv
from datetime import datetime
import pickle
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from maze_env_goal import MazeEnvGoal

WINDOW_SIZE = 5
TOTAL_TIMESTEPS = 1000000
SEED = 42


with open("../datasets/train_mazes5000.pkl", "rb") as f:
    train_mazes = pickle.load(f)
with open("../datasets/test_mazes500.pkl", "rb") as f:
    test_mazes = pickle.load(f)

print(f"Loaded {len(train_mazes)} train mazes, {len(test_mazes)} test mazes")



env = MazeEnvGoal(window_size=WINDOW_SIZE, maze_set=train_mazes, reward_shaping = True)
env = Monitor(env)


model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=2_000,
    exploration_fraction=0.4,       # decay epsilon over 40% of training
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    seed=SEED,
)

print(f"\nTraining DQN for {TOTAL_TIMESTEPS:,} timesteps on {len(train_mazes)} mazes...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

model.save(f"../models/dqn_generalize_w{WINDOW_SIZE}_5kmazes")
print("\nModel saved to ../models/dqn_generalize")


def evaluate_on_set(model, maze_set, window_size, label, max_steps=200):
    """Run the trained agent on each maze in a set; report success and steps."""
    successes = 0
    total_steps = []
    optimal_ratios = []

    for maze_data in maze_set:
        eval_env = MazeEnvGoal(window_size=window_size, maze_set=[maze_data])
        obs, info = eval_env.reset()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            steps += 1
        if terminated:
            successes += 1
            total_steps.append(steps)
            optimal_ratios.append(steps / maze_data["optimal_len"])

    n = len(maze_set)
    success_rate = successes / n * 100
    avg_ratio = np.mean(optimal_ratios) if optimal_ratios else 0.0

    print(f"\n--- {label} ({n} mazes) ---")
    print(f"Success rate: {successes}/{n} ({success_rate:.1f}%)")
    if total_steps:
        print(f"Avg steps (successful): {np.mean(total_steps):.1f}")
        print(f"Avg ratio to optimal:   {avg_ratio:.2f}x (1.0 = perfect)")

    return success_rate, avg_ratio   


# Train set: how well it does on mazes it trained on
train_acc, train_ratio = evaluate_on_set(model, train_mazes, WINDOW_SIZE, "TRAIN set")
test_acc, test_ratio = evaluate_on_set(model, test_mazes, WINDOW_SIZE, "TEST set (UNSEEN)")


print("\n" + "=" * 50)
print("GENERALIZATION SUMMARY")
print("=" * 50)
print(f"Train accuracy: {train_acc:.1f}%")
print(f"Test accuracy:  {test_acc:.1f}%  <- generalization")
print(f"Generalization gap: {train_acc - test_acc:.1f}%")
print("=" * 50)

print("\n✓ Generalization training complete!")

RESULTS_DIR = os.path.join("..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = os.path.join(RESULTS_DIR, "generalization_results.csv")

# Write header only if file doesn't exist yet
file_exists = os.path.isfile(results_path)

with open(results_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "timestamp", "window_size", "total_timesteps", "net_arch",
            "train_acc", "test_acc", "gen_gap",
            "train_ratio", "test_ratio", "seed"
        ])
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        WINDOW_SIZE, TOTAL_TIMESTEPS, "256x256",
        f"{train_acc:.1f}", f"{test_acc:.1f}", f"{train_acc - test_acc:.1f}",
        f"{train_ratio:.3f}", f"{test_ratio:.3f}", SEED
    ])

print(f"\nResults appended to: {results_path}")
