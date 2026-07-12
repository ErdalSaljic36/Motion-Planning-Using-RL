import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

CELL_SIZE = 50 #50 pixels for each cell
WINDOW_PADDING = 0
FPS = 10

#RGB notation
COLOR_EMPTY = (255, 255, 255) #White
COLOR_WALL = (40, 40, 40) #Dark gray
COLOR_AGENT = (255, 0, 0) #Red
COLOR_GOAL = (0, 255, 0) #Green
COLOR_MAZE = (200, 200, 200) #Gray edges of the maze




class MazeEnv(gym.Env): #Class gym.Env is parent class of MazeEnv
    """
    Grid-world environment represented as 2D Numpy array: 
        - 0 means empty cell
        - 1 means wall
    """
    #Matadata needed for rendering in later stages
    metadata={"render_modes": ["human"], "render_fps":10}

    #Actions:
    ACTION_UP=0
    ACTION_RIGHT=1
    ACTION_LEFT=2
    ACTION_DOWN=3

    def __init__(self, max_steps: int=200, render_mode: str | None = None, maze_set: list | None=None, reward_shaping: bool = False):
        super().__init__()

        #If we want to use randomly generated maze sets for neural network training
        self.maze_set=maze_set
        self.reward_shaping = reward_shaping
        self.shaping_scale = 0.1              # reward for moving toward the goal
        self.shaping_scale_away = 0.033       # weaker penalty for moving away (3x)

        #Hardcoded maze layout
        self.maze = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=np.int32)


        if self.maze_set is not None:
            self._load_maze(self.maze_set[0])
        else:
            self.maze_size = self.maze.shape[0]
            self.start_pos = np.array([0, 0], dtype=np.int32)
            self.goal_pos = np.array([9, 9], dtype=np.int32)
        #Our atributes
        self.max_steps=max_steps
        self.current_step=0
        self.agent_pos=None

        #Mandatory atributes from gym.Env
        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Box (low=0, high=self.maze_size-1, shape=(2, ), dtype=np.int32)

        #Rendering for visualization
        self.render_mode = render_mode
        self.window = None
        self. clock= None

    def _load_maze(self, maze_data: dict):
        self.maze = maze_data["maze"].copy()
        self.maze_size=self.maze.shape[0]
        self.start_pos=np.array(maze_data["start"], dtype=np.int32)
        self.goal_pos = np.array(maze_data["goal"], dtype=np.int32)
        self.optimal_len = maze_data.get("optimal_len", None)


    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.maze_set is not None:
            idx = self.np_random.integers(0, len(self.maze_set))
            self._load_maze(self.maze_set[idx])

        self.agent_pos=self.start_pos.copy()
        self.current_step=0

        observation=self._get_observation()
        info={}

        return observation, info
    
    #Step function prepared for advanced algorithms to use in Gymnasium API (returns 5 values)
    def step(self, action: int):
        self.current_step += 1

        next_pos = self.agent_pos.copy()
        if action == self.ACTION_UP:
            next_pos[0] -= 1
        elif action == self.ACTION_DOWN:
            next_pos[0] += 1
        elif action == self.ACTION_RIGHT:
            next_pos[1] += 1
        elif action == self.ACTION_LEFT:
            next_pos[1] -= 1
        else:
            raise ValueError(f"Invalid action: {action}")

        if self._is_valid_position(next_pos):
            if self.reward_shaping:
                old_dist = np.abs(self.agent_pos - self.goal_pos).sum()
                new_dist = np.abs(next_pos - self.goal_pos).sum()
                delta = old_dist - new_dist

                # Asymmetric shaping --> closer = +0.1, moving away = -0.033
                if delta > 0:
                    shaping = delta * self.shaping_scale         # approaching: +0.1
                else:
                    shaping = delta * self.shaping_scale_away    # moving away: -0.033

                reward = -0.01 + shaping
            else:
                reward = -0.01

                self.agent_pos = next_pos
        else:
            reward = -0.

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        if terminated:
            reward = 1.0

        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info
    
    
    #Setting this method as private because it is used only in reset() and step() funcions
    def _get_observation(self) -> np.ndarray:
        return self.agent_pos.copy()
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        row, col = pos[0], pos[1]

        if row < 0 or row >= self.maze_size:
            return False
        if col < 0 or col >= self.maze_size:
            return False
        if self.maze[row, col] == 1:
            return False
        return True
    


    def render(self):
        if self.render_mode != "human":
            return
        if self.window==None:
            pygame.init()
            pygame.display.set_caption("MazeEnv - Motion planning using RL")
            window_size=self.maze_size * CELL_SIZE
            self.window = pygame.display.set_mode((window_size, window_size))
            self.clock = pygame.time.Clock()

        
        #Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        #Backgorund (cells)
        self.window.fill(COLOR_EMPTY)

        #Walls
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                if self.maze[row, col] == 1:
                    rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.window, COLOR_WALL, rect)
        
        #Goal
        goal_row, goal_col = self.goal_pos[0], self.goal_pos[1]
        goal_rect = pygame.Rect(goal_col * CELL_SIZE + 5, goal_row * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.rect(self.window, COLOR_GOAL, goal_rect)
        
        # Draw agent (circle in the center of its cell)
        agent_row, agent_col = self.agent_pos[0], self.agent_pos[1]
        agent_center = (agent_col * CELL_SIZE + CELL_SIZE // 2, agent_row * CELL_SIZE + CELL_SIZE // 2)
        agent_radius = CELL_SIZE // 3
        pygame.draw.circle(self.window, COLOR_AGENT, agent_center, agent_radius)


        for i in range(self.maze_size + 1):
            # Vertical lines
            pygame.draw.line(self.window, COLOR_MAZE, (i * CELL_SIZE, 0), (i * CELL_SIZE, self.maze_size * CELL_SIZE), 1)
            # Horizontal lines
            pygame.draw.line(self.window, COLOR_MAZE, (0, i * CELL_SIZE), (self.maze_size * CELL_SIZE, i * CELL_SIZE), 1)
        self._render_overlay()
        pygame.display.flip()
        self.clock.tick(FPS)



    def _render_overlay(self):
        #Method created for the subclass to override it
        pass
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    


        




