#Visualize the learned Q-Learning policy.

import numpy as np
import matplotlib.pyplot as plt
import os
from maze_env import MazeEnv

MODELS_DIR = os.path.join("..", "models")
FIGURES_DIR = os.path.join("..", "figures")
Q_TABLE_PATH = os.path.join(MODELS_DIR, "q_table.npy")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Action convention (must match MazeEnv!):
# UP=0, RIGHT=1, LEFT=2, DOWN=3
# Arrow directions (dx, dy) in plot coordinates.
# Note: in imshow, y increases downward, so UP means dy = -1.
ACTION_ARROWS = {
    0: (0, -1),   # UP
    1: (1, 0),    # RIGHT
    2: (-1, 0),   # LEFT
    3: (0, 1),    # DOWN
}


env = MazeEnv()
q_table = np.load(Q_TABLE_PATH)
maze = env.maze
maze_size = env.maze_size
goal_pos = env.goal_pos
start_pos = env.start_pos

print(f"Loaded Q-table with shape {q_table.shape}")


# For each cell take the maximum Q-value across all actions
max_q_values = np.max(q_table, axis=2)  

# Mask walls so they appear black
masked_q = np.ma.masked_where(maze == 1, max_q_values)

fig, ax = plt.subplots(figsize=(9, 9))
cmap = plt.cm.viridis
cmap.set_bad(color="black")  

im = ax.imshow(masked_q, cmap=cmap, interpolation="nearest")

# Give each cell its max Q-value
for row in range(maze_size):
    for col in range(maze_size):
        if maze[row, col] == 0:  # only empty cells
            value = max_q_values[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    color="white", fontsize=7)

# Mark start and goal
ax.scatter(start_pos[1], start_pos[0], marker="s", s=200,
           edgecolor="cyan", facecolor="none", linewidth=3, label="Start")
ax.scatter(goal_pos[1], goal_pos[0], marker="*", s=400,
           edgecolor="red", facecolor="red", label="Goal")

ax.set_title("Q-Learning: Max Q-Value per Cell (State Value)")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.legend(loc="upper right")
plt.colorbar(im, ax=ax, label="Max Q-value", fraction=0.046, pad=0.04)
plt.tight_layout()

path = os.path.join(FIGURES_DIR, "qlearning_value_heatmap.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")

fig, ax = plt.subplots(figsize=(9, 9))

# Draw maze background: walls black, empty white
display_maze = np.ones((maze_size, maze_size, 3))  
display_maze[maze == 1] = [0.2, 0.2, 0.2]          
ax.imshow(display_maze, interpolation="nearest")

# Draw an arrow in each empty cell showing the best action
for row in range(maze_size):
    for col in range(maze_size):
        if maze[row, col] == 1:
            continue  # skip walls
        if (row, col) == tuple(goal_pos):
            continue  # skip goal

        best_action = int(np.argmax(q_table[row, col]))
        dx, dy = ACTION_ARROWS[best_action]

        # Arrow centered in the cell, scaled to fit
        ax.arrow(
            col, row,             
            dx * 0.3, dy * 0.3,   
            head_width=0.2,
            head_length=0.2,
            fc="darkblue",
            ec="darkblue",
        )

# Mark start and goal
ax.scatter(start_pos[1], start_pos[0], marker="s", s=200,
           edgecolor="cyan", facecolor="none", linewidth=3, label="Start")
ax.scatter(goal_pos[1], goal_pos[0], marker="*", s=400,
           edgecolor="green", facecolor="green", label="Goal")

ax.set_title("Q-Learning: Learned Policy (Best Action per Cell)")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.set_xticks(range(maze_size))
ax.set_yticks(range(maze_size))
ax.legend(loc="upper right")
plt.tight_layout()

path = os.path.join(FIGURES_DIR, "qlearning_policy_arrows.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Figure saved: {path}")


print("\n Policy visualization complete!")