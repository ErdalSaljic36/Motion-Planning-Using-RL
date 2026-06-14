#Tabular Q-Learning 

import numpy as np


class QLearningAgent:
    def __init__(self, maze_size : int, num_actions : int=4, alpha : float = 0.1, gamma : float = 0.99, eps_start : float=1.0, eps_min : float=0.05, eps_decay:float=0.995, seed:int | None=None):
        
        self.maze_size=maze_size
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay= eps_decay
        
        #Inital table of Q-values is empty ; for every position (=state) we need 4 values (corresponding to 4 actions)
        self.q_table = np.zeros((maze_size, maze_size, num_actions))

        #For reproducibility ; every instance has it's own seed
        self.rng = np.random.default_rng(seed)

    def choose_action(self, observation : np.ndarray) -> int:
        #Exploration
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.num_actions))
        
        #Exploitation
        row, col = observation[0], observation[1]
        q_values = self.q_table[row, col]

        #In case there are multiple actions with the same best value, pick randomly amongst them
        max_q = np.max(q_values)
        best_action = np.where(q_values == max_q)[0]
        return int(self.rng.choice(best_action))
    
    #Updates the Q-Learning table
    def update(self, observation, action, reward, next_observation, terminated):
        row, col = int(observation[0]), int(observation[1])
        next_row, next_col = int(next_observation[0]), int(next_observation[1])
        action = int(action)

        # Current estimate
        current_q = self.q_table[row, col, action]

        # Best possible future value from the next state
        if terminated:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_row, next_col])

        # TD target and TD error
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q

        # Update
        self.q_table[row, col, action] = current_q + self.alpha * td_error

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay) #Lowers eps by multiplying it by 0.995 each episode but doesn't allow it to fall below 0.05

 









