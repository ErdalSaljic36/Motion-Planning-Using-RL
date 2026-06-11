#Experiment 1: Effect of epsilon dacay rate on learning

import numpy as np
import matplotlib.pyplot as plt
import os
from experiment_utils import train_agent, moving_average

FIGURES_DIR =os.path.join("..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

NUM_EPISODES = 5000
SEED = 42
SMOOTH_WINDOW = 100

#Three variants of decay rate
DECAY_VARIANTS = {
    "Fast (0.99)": 0.99,        #~300 episodes to reach eps_min
    "Medium (0.995)": 0.995,    #~600 episodes to reach eps_min
    "Slow (0.999)": 0.999,      #~3000 episodes to reach eps_min
}

COLORS = {
    "Fast (0.99)": "tomato",
    "Medium (0.995)": "steelblue",
    "Slow (0.999)": "seagreen",
}


print("Experiment 1 - Epsilon decay")

results = {}

for label, decay in DECAY_VARIANTS.items():
    print(f"Training with eps_decay = {decay} ({label})")
    results[label]= train_agent(eps_start = 1.0, eps_min=0.05, eps_decay=decay, num_episodes=NUM_EPISODES, seed=SEED)

print("Training complete, generating plots...")

#Fig 1
fig, ax = plt.subplots(figsize=(12, 6))

for label in DECAY_VARIANTS:
    rewards = results[label]["episode_rewards"]
    smoothed = moving_average(rewards, window=SMOOTH_WINDOW)
    ax.plot(smoothed, label=label, color=COLORS[label], linewidth=2)

ax.set_xlabel("Episode")
ax.set_ylabel(f"Total reward (moving avg, window={SMOOTH_WINDOW})")
ax.set_title("Experiment 1: Reward Curves for Different Epsilon Decay Rates")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "exp1_reward_curves.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")


#Fig 2
fig, ax = plt.subplots(figsize=(12, 6))

for label in DECAY_VARIANTS:
    successes = results[label]["episode_successes"].astype(float)
    smoothed = moving_average(successes, window=SMOOTH_WINDOW) * 100
    ax.plot(smoothed, label=label, color=COLORS[label], linewidth=2)

ax.set_xlabel("Episode")
ax.set_ylabel(f"Success rate (%, rolling {SMOOTH_WINDOW})")
ax.set_title("Experiment 1: Success Rate for Different Epsilon Decay Rates")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "exp1_success_curves.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")


#Fig 3
fig, ax = plt.subplots(figsize=(12, 6))

for label in DECAY_VARIANTS:
    eps_history = results[label]["epsilon_history"]
    ax.plot(eps_history, label=label, color=COLORS[label], linewidth=2)

ax.set_xlabel("Episode")
ax.set_ylabel("Epsilon (exploration rate)")
ax.set_title("Experiment 1: Epsilon Decay Schedules")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "exp1_epsilon_schedules.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")


#Summary table
print("\n" + "=" * 55)
print("SUMMARY (last 500 episodes)")
print("=" * 55)
print(f"{'Variant':<18}{'Success %':<12}{'Mean reward':<14}{'Mean length'}")
print("-" * 55)
for label in DECAY_VARIANTS:
    succ = results[label]["episode_successes"][-500:].mean() * 100
    rew = results[label]["episode_rewards"][-500:].mean()
    length = results[label]["episode_lengths"][-500:].mean()
    print(f"{label:<18}{succ:<12.1f}{rew:<14.3f}{length:.1f}")
print("=" * 55)

print("\n✓ Experiment 1 complete!")














