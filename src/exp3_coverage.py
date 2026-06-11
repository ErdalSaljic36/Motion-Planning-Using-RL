import numpy as np
import matplotlib.pyplot as plt
import os
from experiment_utils import train_agent, moving_average
from maze_env import MazeEnv

FIGURES_DIR = os.path.join("..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

NUM_EPISODES = 5000
SEED = 42

STRATEGIES = {
    "Low fixed (0.05)": (0.05, 0.05, 1.0),      #(eps_start, eps_min, eps_decay)
    "High fixed (0.5)": (0.5, 0.05, 1.0), 
    "Decay (1.0 -> 0.05)": (1.0, 0.05, 0.995),
    "Random (eps=1.0)": (1.0, 1.0, 1.0)
}

print("Experiment 3: Visitation map")

env_info = MazeEnv()
maze=env_info.maze      #To get the np.array 
total_walkable = int(np.sum(maze==0))       #Count all the empty cells
print(f"Total walkable cells: {total_walkable}")

results = {}
for label, (eps_start, eps_min, eps_dcay) in STRATEGIES.items():
    print(f"Training: {label}")
    results[label] = train_agent(eps_start=eps_start, eps_min=eps_min, eps_decay=eps_dcay, num_episodes=NUM_EPISODES, seed=SEED, track_visitation=True)

print("Training complete, generating figures...")


#Figure: 2x2 grid of visitation heatmaps (heatmap for each strategy)

fig, axes = plt.subplots(2, 2, figsize=(14, 13))
axes = axes.flatten()

cmap = plt.cm.hot
cmap.set_bad(color="black")  # walls

for idx, label in enumerate(STRATEGIES):
    ax = axes[idx]
    visitation = results[label]["visitation_map"]

    # Mask walls so they appear black
    masked = np.ma.masked_where(maze == 1, visitation)

    # Use log scale for color (visitation spans huge range)
    # Add 1 to avoid log(0)
    im = ax.imshow(np.log10(masked + 1), cmap=cmap, interpolation="nearest")

    # Mark start and goal
    ax.scatter(env_info.start_pos[1], env_info.start_pos[0], marker="s",
               s=150, edgecolor="cyan", facecolor="none", linewidth=2.5)
    ax.scatter(env_info.goal_pos[1], env_info.goal_pos[0], marker="*",
               s=250, edgecolor="lime", facecolor="lime")

    # Count how many walkable cells were visited at least once
    visited_cells = int(np.sum((visitation > 0) & (maze == 0)))
    coverage_pct = visited_cells / total_walkable * 100

    ax.set_title(f"{label}\nCoverage: {visited_cells}/{total_walkable} "
                 f"cells ({coverage_pct:.1f}%)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, label="log10(visits + 1)",
                 fraction=0.046, pad=0.04)

fig.suptitle("Experiment 3: State-Space Coverage by Exploration Strategy",
             fontsize=15, y=1.00)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "exp3_coverage_heatmaps.png")
plt.savefig(path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Figure saved: {path}")


#Summary table
print("\n" + "=" * 60)
print("COVERAGE SUMMARY")
print("=" * 60)
print(f"{'Strategy':<24}{'Coverage':<14}{'Success %':<12}{'Mean length'}")
print("-" * 60)
for label in STRATEGIES:
    visitation = results[label]["visitation_map"]
    visited_cells = int(np.sum((visitation > 0) & (maze == 0)))
    coverage_pct = visited_cells / total_walkable * 100
    succ = results[label]["episode_successes"][-500:].mean() * 100
    length = results[label]["episode_lengths"][-500:].mean()
    print(f"{label:<24}{coverage_pct:<14.1f}{succ:<12.1f}{length:.1f}")
print("=" * 60)

print("\n✓ Experiment 3 complete!")





