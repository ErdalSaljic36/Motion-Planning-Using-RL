import os
import pickle
from maze_generator import generate_maze_set


MAZE_SIZE = 10
WALL_DENSITY = 0.30
MIN_PATH_LENGTH = 4

NUM_TRAIN = 150
NUM_TEST = 30

TRAIN_SEED = 42
TEST_SEED = 999

OUTPUT_DIR = os.path.join("..", "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)


print(f"Generating {NUM_TRAIN} training sets...")
train_mazes = generate_maze_set(count = NUM_TRAIN, size = MAZE_SIZE, wall_density=WALL_DENSITY, seed= TRAIN_SEED, min_path_length=MIN_PATH_LENGTH)

print(f"Generating {NUM_TEST} test sets...")
test_mazes = generate_maze_set(count = NUM_TEST, size = MAZE_SIZE, wall_density=WALL_DENSITY, seed= TEST_SEED, min_path_length=MIN_PATH_LENGTH)


train_path = os.path.join(OUTPUT_DIR, "train_mazes.pkl")
test_path = os.path.join(OUTPUT_DIR, "test_mazes.pkl")


with open(train_path, "wb") as f:
    pickle.dump(train_mazes, f)
with open(test_path, "wb") as f:
    pickle.dump(test_mazes, f)


print(f"\nSaved {NUM_TRAIN} training mazes to: {train_path}")
print(f"\nSaved {NUM_TEST} test mazes to: {test_path}")

train_lengths = [m["optimal_len"] for m in train_mazes]
test_lengths = [m["optimal_len"] for m in test_mazes]

print("\n--- Dataset statistics ---")
print(f"Train optimal path lengths: "
      f"min={min(train_lengths)}, max={max(train_lengths)}, "
      f"avg={sum(train_lengths)/len(train_lengths):.1f}")
print(f"Test optimal path lengths:  "
      f"min={min(test_lengths)}, max={max(test_lengths)}, "
      f"avg={sum(test_lengths)/len(test_lengths):.1f}")
print("\n✓ Datasets generated!")
