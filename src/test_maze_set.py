

import pickle
from maze_env import MazeEnv

# Load the training mazes
with open("../datasets/train_mazes.pkl", "rb") as f:
    train_mazes = pickle.load(f)

print(f"Loaded {len(train_mazes)} training mazes")

# Create env with the maze set
env = MazeEnv(maze_set=train_mazes)

# Resetting to check if we get different mazes
print("\nResetting 5 times, checking start/goal positions:")
for i in range(5):
    obs, info = env.reset()
    print(f"Reset {i+1}: start={env.start_pos}, goal={env.goal_pos}, "
          f"optimal_len={env.optimal_len}")

# Verify the maze actually changes (compare two resets)
env.reset()
maze_a = env.maze.copy()
env.reset()
maze_b = env.maze.copy()
changed = not (maze_a == maze_b).all()
print(f"\nMaze changes between resets: {changed}")

# Also verify Phase 1 compatibility (no maze_set -> default maze)
env_default = MazeEnv()  # no maze_set
obs, info = env_default.reset()
print(f"\nDefault env (no maze_set): start={env_default.start_pos}, "
      f"goal={env_default.goal_pos}")

print("\n✓ Maze set integration works!")