import os
import csv
import pickle
from datetime import datetime

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from maze_env_goal import MazeEnvGoal

WINDOW_SIZE = 3
TOTAL_TIMESTEPS = 1_000_000
REWARD_SHAPING = True

SEEDS = [42, 123, 999]        # three training seeds

TRAIN_PATH = "../datasets/train_mazes5000.pkl"
TEST_PATH = "../datasets/test_mazes500.pkl"
MODEL_DIR = "../models"
RESULTS_DIR = "../results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(TRAIN_PATH, "rb") as f:
    train_mazes = pickle.load(f)
with open(TEST_PATH, "rb") as f:
    test_mazes = pickle.load(f)

print(f"Loaded {len(train_mazes)} train mazes, {len(test_mazes)} test mazes")
print(f"Running {len(SEEDS)} seeds: {SEEDS}\n")

def evaluate_on_set(model, maze_set, window_size, label):
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

    print(f"  {label}: {successes}/{n} ({success_rate:.1f}%), "
          f"ratio {avg_ratio:.2f}x")

    return success_rate, avg_ratio

def run_seed(seed):
    print(f"\n{'=' * 55}")
    print(f"SEED {seed}")
    print(f"{'=' * 55}")

    env = MazeEnvGoal(window_size=WINDOW_SIZE, maze_set=train_mazes,
                      reward_shaping=REWARD_SHAPING)
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
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,                      # quiet — we only want the final numbers
        seed=seed,
    )

    print(f"Training ({TOTAL_TIMESTEPS:,} steps)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save(os.path.join(MODEL_DIR, f"dqn_w{WINDOW_SIZE}_5000mazes_seed{seed}"))

    print("Evaluating...")
    train_acc, train_ratio = evaluate_on_set(model, train_mazes, WINDOW_SIZE,
                                             "TRAIN")
    test_acc, test_ratio = evaluate_on_set(model, test_mazes, WINDOW_SIZE,
                                           "TEST ")

    return {
        "seed": seed,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": train_acc - test_acc,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
    }

results = []
for seed in SEEDS:
    results.append(run_seed(seed))

train_accs = [r["train_acc"] for r in results]
test_accs = [r["test_acc"] for r in results]
gaps = [r["gap"] for r in results]

print("\n" + "=" * 55)
print("MULTI-SEED SUMMARY (DQN, 3x3, 5000 mazes)")
print("=" * 55)
print(f"{'Seed':<8}{'Train %':<12}{'Test %':<12}{'Gap %'}")
print("-" * 55)
for r in results:
    print(f"{r['seed']:<8}{r['train_acc']:<12.1f}{r['test_acc']:<12.1f}"
          f"{r['gap']:.1f}")
print("-" * 55)
print(f"{'MEAN':<8}{np.mean(train_accs):<12.1f}{np.mean(test_accs):<12.1f}"
      f"{np.mean(gaps):.1f}")
print(f"{'STD':<8}{np.std(train_accs):<12.1f}{np.std(test_accs):<12.1f}"
      f"{np.std(gaps):.1f}")
print("=" * 55)
print(f"\nTest accuracy: {np.mean(test_accs):.1f}% ± {np.std(test_accs):.1f}%")

results_path = os.path.join(RESULTS_DIR, "multiseed_results.csv")
file_exists = os.path.isfile(results_path)

with open(results_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "timestamp", "algorithm", "window_size", "num_mazes",
            "total_timesteps", "seed",
            "train_acc", "test_acc", "gen_gap", "train_ratio", "test_ratio"
        ])
    for r in results:
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            "DQN", WINDOW_SIZE, len(train_mazes), TOTAL_TIMESTEPS, r["seed"],
            f"{r['train_acc']:.1f}", f"{r['test_acc']:.1f}", f"{r['gap']:.1f}",
            f"{r['train_ratio']:.3f}", f"{r['test_ratio']:.3f}"
        ])

print(f"\nResults appended to: {results_path}")
print("\n✓ Multi-seed run complete!")