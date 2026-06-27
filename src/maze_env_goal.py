"""
MazeEnvGoal: goal-conditioned POMDP maze environment.

Extends MazeEnvWindow by adding a normalized goal signal to the observation.
The observation is a flat vector: [flattened local window] + [normalized
relative vector to goal]. This enables goal-conditioned generalization across
mazes and (with normalization) across maze sizes.
"""

import numpy as np
from gymnasium import spaces
from maze_env_window import MazeEnvWindow


class MazeEnvGoal(MazeEnvWindow):
    def __init__(self, max_steps: int = 200, render_mode: str | None = None,
                 window_size: int = 3, maze_set: list | None = None):
        super().__init__(max_steps=max_steps, render_mode=render_mode,
                         window_size=window_size, maze_set=maze_set)

        window_cells = window_size * window_size
        obs_dim = window_cells + 2

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        window = self._get_local_window().flatten().astype(np.float32)

        #Normalized relative vector to goal
        drow = (self.goal_pos[0] - self.agent_pos[0]) / self.maze_size
        dcol = (self.goal_pos[1] - self.agent_pos[1]) / self.maze_size
        goal_signal = np.array([drow, dcol], dtype=np.float32)

        #Concatenate into one flat vector
        observation = np.concatenate([window, goal_signal])
        return observation