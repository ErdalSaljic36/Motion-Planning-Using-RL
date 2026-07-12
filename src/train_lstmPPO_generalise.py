import os
import csv
import pickle
from datetime import datetime

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

from maze_env_goal import MazeEnvGoal


ON_COLAB = os.path.exists("/content")

if ON_COLAB:
    BASE = "/content/drive/MyDrive/Motion planning using RL"
else:
    BASE = ".."   

TRAIN_PATH = os.path.join(BASE, "datasets", "train_mazes5000.pkl")
TEST_PATH = os.path.join(BASE, "datasets", "test_mazes500.pkl")
MODEL_DIR = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Running on {'COLAB' if ON_COLAB else 'LOCAL'} | base path: {BASE}")


WINDOW_SIZE = 3
TOTAL_TIMESTEPS = 1_000_000
REWARD_SHAPING = True
SEED = 123

with open(TRAIN_PATH, "rb") as f:
    train_mazes = pickle.load(f)
with open(TEST_PATH, "rb") as f:
    test_mazes = pickle.load(f)

print(f"Loaded {len(train_mazes)} train mazes, {len(test_mazes)} test mazes")


env = MazeEnvGoal(window_size=WINDOW_SIZE, maze_set=train_mazes,
                  reward_shaping=REWARD_SHAPING)
env = Monitor(env)

model = RecurrentPPO(
    "MlpLstmPolicy",             
    env,
    learning_rate=3e-4,
    n_steps=1024,                 
    batch_size=128,              
    n_epochs=10,                 
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,               
    ent_coef=0.01,                
    policy_kwargs=dict(
        net_arch=[256, 256],      
        lstm_hidden_size=128,     
        n_lstm_layers=1,
    ),
    verbose=1,
    seed=SEED,
)

print(f"\nTraining Recurrent PPO for {TOTAL_TIMESTEPS:,} timesteps "
      f"on {len(train_mazes)} mazes...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

model_path = os.path.join(
    MODEL_DIR, f"lstm_ppo_w{WINDOW_SIZE}_{len(train_mazes)}mazes_seed{SEED}"
)
model.save(model_path)
print(f"\nModel saved to: {model_path}")

def evaluate_on_set(model, maze_set, window_size, label):
    
    successes = 0
    total_steps = []
    optimal_ratios = []

    for maze_data in maze_set:
        eval_env = MazeEnvGoal(window_size=window_size, maze_set=[maze_data])
        obs, info = eval_env.reset()

        lstm_state = None        # memory starts empty
        episode_start = True     # signals the LSTM to reset its state

        terminated = truncated = False
        steps = 0

        while not (terminated or truncated):
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=np.array([episode_start]),
                deterministic=True,
            )
            episode_start = False   # only the first step starts an episode

            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            steps += 1

        if terminated:              # reached the goal
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


train_acc, train_ratio = evaluate_on_set(model, train_mazes, WINDOW_SIZE,
                                         "TRAIN set")
test_acc, test_ratio = evaluate_on_set(model, test_mazes, WINDOW_SIZE,
                                       "TEST set (UNSEEN)")


print("\n" + "=" * 55)
print("GENERALIZATION SUMMARY (Recurrent PPO / LSTM)")
print("=" * 55)
print(f"Train accuracy: {train_acc:.1f}%")
print(f"Test accuracy:  {test_acc:.1f}%  <- generalization")
print(f"Generalization gap: {train_acc - test_acc:.1f}%")
print(f"\nDQN baseline (same setup): 28.9% train / 30.2% test")
print("=" * 55)


results_path = os.path.join(RESULTS_DIR, "algorithm_comparison.csv")
file_exists = os.path.isfile(results_path)

with open(results_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "timestamp", "algorithm", "window_size", "num_mazes",
            "total_timesteps", "reward_shaping",
            "train_acc", "test_acc", "gen_gap",
            "train_ratio", "test_ratio", "seed"
        ])
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "RecurrentPPO-LSTM", WINDOW_SIZE, len(train_mazes),
        TOTAL_TIMESTEPS, REWARD_SHAPING,
        f"{train_acc:.1f}", f"{test_acc:.1f}", f"{train_acc - test_acc:.1f}",
        f"{train_ratio:.3f}", f"{test_ratio:.3f}", SEED
    ])

print(f"\nResults appended to: {results_path}")
print("\n✓ Recurrent PPO training complete!")