import numpy as np
import gymnasium as gym
from gymnasium import spaces

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

    def __init__(self, max_steps: int=200):
        super().__init__()

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

        #Our atributes
        self.maze_size=self.maze.shape[0]
        self.start_pos=np.array([0, 0], dtype=np.int32)
        self.goal_pos=np.array([9, 9], dtype=np.int32)
        self.max_steps=max_steps
        self.current_step=0
        self.agent_pos=None

        #Mandatory atributes from gym.Env
        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Box (low=0, high=self.maze_size-1, shape=(2, ), dtype=np.int32)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos=self.start_pos.copy()
        self.current_step=0

        observation=self._get_observation()
        info={}

        return observation, info
    
    #Step function prepared for advanced algorithms to use in Gymnasium API (returns 5 values)
    def step(self, action: int):
        
        self.current_step +=1

        next_pos=self.agent_pos.copy()
        if action == self.ACTION_UP:
            next_pos[0] -=1
        elif action == self.ACTION_DOWN:
            next_pos[0] +=1
        elif action == self.ACTION_RIGHT:
            next_pos[1] +=1
        elif action == self.ACTION_LEFT:
            next_pos[1] -=1
        else:
            raise ValueError(f"Invalid action: {action}")
        
        if self._is_valid_position(next_pos):
            self.agent_pos=next_pos
            reward = -0.01
        else:
            reward =-0.1
        
        terminated=np.array_equal(self.agent_pos, self.goal_pos)
        if terminated:
            reward =1

        truncated=self.current_step >= self.max_steps

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
        pass
    def close(self):
        pass

    


        




