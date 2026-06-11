from maze_env import MazeEnv, CELL_SIZE
import numpy as np
from gymnasium import spaces
import pygame 

COLOR_FOV_FRAME = (255, 0, 0)
FOV_FRAME_THICKNESS = 4


class MazeEnvWindow(MazeEnv):
    """
    Partial Observation Markov Decision Porcess
    The observation is now 3x3 matrix, with agent being at the position [1][1]. 1 indicates a wall and 0 indicates an empty cell
    The cells outside the bounderies of the maze are interpreted as a wall
    The 3x3 grid observation will later be flattened into 9D array for easeir manipulation
    """
    def __init__(self, max_steps: int= 200, render_mode: str | None = None):
        super().__init__(max_steps=max_steps, render_mode=render_mode)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.int32)

    def _get_observation(self) -> np.ndarray:
        return self._get_local_window()
    
    def _get_local_window(self) -> np.ndarray:

        window = np.ones((3, 3), dtype = np.int32)
        row = int(self.agent_pos[0])
        col = int(self.agent_pos[1])

        for i in range (-1, 2):
            for j in range (-1, 2):
                r = row +i
                c = col+j
                if 0 <= r < self.maze_size and 0 <= c < self.maze_size:
                    window[i+1, j+1] = self.maze[r, c]
        return window
    
    def _render_overlay(self):
        if self.render_mode != "human" or self.render_mode == None:
            return 
        
        row = int(self.agent_pos[0])
        col = int(self.agent_pos[1])

        x = (col - 1) * CELL_SIZE
        y = (row - 1) * CELL_SIZE
        size = 3 * CELL_SIZE

        frame_rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.window, COLOR_FOV_FRAME, frame_rect, FOV_FRAME_THICKNESS)










