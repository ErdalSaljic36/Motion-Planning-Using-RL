import numpy as np
import os
from experiment_utils import train_agent, moving_average
import matplotlib.pyplot as plt


FIGURES_DIR= os.path.join("..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

NUM_EPISODES = 5000
MAX_STEPS = 200
SEED = 42
SMOOTH_WINDOW = 100

EPSILON_VARIANTS = {
    "eps = 0.01": 0.01,
    "eps = 0.1": 0.1,
    "eps = 0.5": 0.5,
    "eps = 1.0": 1.0
}

COLORS = {
    "eps = 0.01": "seagreen",
    "eps = 0.1": "steelblue",
    "eps = 0.5": "darkorange",
    "eps = 1.0": "tomato",
}

print("Experiment 2 - Fixed epsilon (decay = 1.0)")

results = {}

for label, epsilon in EPSILON_VARIANTS.items():
    print(f"Training with fixed epsilon at {epsilon}")
    results[label] = train_agent(eps_start= epsilon, eps_min= epsilon, eps_decay= 1.0, num_episodes= NUM_EPISODES, seed=SEED)

print("Training complete, generating figures...")

#Figure 1
fig, ax = plt.subplots(figsize=(12, 6))
for label in EPSILON_VARIANTS:
    rewards = results[label]["episode_rewards"]
    smoothed = moving_average(rewards, window=SMOOTH_WINDOW)
    ax.plot(smoothed, label=label, color=COLORS[label], linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel(f"Total reward (moving avg, window={SMOOTH_WINDOW})")
ax.set_title("Experiment 2: Reward Curves for Fixed Epsilon Values")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "exp2_reward_curves.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")


#Figure 2
fig, ax = plt.subplots(figsize=(12, 6))
for label in EPSILON_VARIANTS:
    successes = results[label]["episode_successes"].astype(float)
    smoothed = moving_average(successes, window=SMOOTH_WINDOW) * 100
    ax.plot(smoothed, label=label, color=COLORS[label], linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel(f"Success rate (%, rolling {SMOOTH_WINDOW})")
ax.set_title("Experiment 2: Success Rate for Fixed Epsilon Values")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "exp2_success_curves.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")

#Summary table

print("\n" + "=" * 55)
print("SUMMARY (last 500 episodes)")
print("=" * 55)
print(f"{'Variant':<22}{'Success %':<12}{'Mean reward':<14}{'Mean length'}")
print("-" * 55)
for label in EPSILON_VARIANTS:
    succ = results[label]["episode_successes"][-500:].mean() * 100
    rew = results[label]["episode_rewards"][-500:].mean()
    length = results[label]["episode_lengths"][-500:].mean()
    print(f"{label:<22}{succ:<12.1f}{rew:<14.3f}{length:.1f}")
print("=" * 55)

print("\n✓ Experiment 2 complete!")





