

import pickle
import pygame
from maze_env_goal import MazeEnvGoal

TRAIN_PATH = "../datasets/train_mazes450.pkl"   
TEST_PATH = "../datasets/test_mazes50.pkl"     
WINDOW_SIZE = 3


def draw_and_save(maze_data, filename):
    #Load one maze into the env, render it, and save a screenshot.
    env = MazeEnvGoal(window_size=WINDOW_SIZE, maze_set=[maze_data],
                      render_mode="human")
    env.reset()
    env.render()

    # Save the current Pygame window surface to a file
    pygame.image.save(env.window, filename)
    print(f"Saved {filename} "
          f"(start={maze_data['start']}, goal={maze_data['goal']}, "
          f"optimal={maze_data['optimal_len']})")
    pygame.time.wait(500)
    env.close()


with open(TRAIN_PATH, "rb") as f:
    train_mazes = pickle.load(f)
with open(TEST_PATH, "rb") as f:
    test_mazes = pickle.load(f)

print(f"Loaded {len(train_mazes)} train, {len(test_mazes)} test mazes\n")

draw_and_save(train_mazes[0], "../figures/sample_train_1.png")
draw_and_save(train_mazes[1], "../figures/sample_train_2.png")
draw_and_save(test_mazes[0], "../figures/sample_test_1.png")
draw_and_save(test_mazes[1], "../figures/sample_test_2.png")

print("\n✓ Done! 4 maze images saved to ../figures/")