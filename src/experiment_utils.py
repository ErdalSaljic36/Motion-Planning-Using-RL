#Shared training function fro running exploration vs. exploitation experiments

from maze_env import MazeEnv
import numpy as np
from q_learning_agent import QLearningAgent


def train_agent(eps_start: float=1.0, eps_min: float=0.5, eps_decay: float=0.995, alpha: float=0.1, gamma:float=0.99, num_episodes: int=5000, max_steps:int=200, seed:int=42, track_visitation:bool=False):
    env = MazeEnv(max_steps=max_steps)
    agent = QLearningAgent(maze_size=env.maze_size, num_actions= env.action_space.n, alpha=alpha, gamma=gamma, eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay, seed=seed)

    episode_rewards=[]
    episode_lengths=[]
    episode_successes=[]
    epsilon_history=[]

    visitation_map = None
    if track_visitation:
        visitation_map = np.zeros((env.maze_size, env.maze_size), dtype = np.int64)

    #Go through all the episodes for statistical analysis
    for episode in range(num_episodes):
        obs, info = env.reset(seed= seed+episode)
        total_reward=0.0
        steps=0

        #Steps in one episode
        for steps in range(max_steps):
            if track_visitation:
                row, col = int(obs[0]), int(obs[1])
                visitation_map[row, col] +=1
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)

            obs = next_obs
            total_reward += reward
            steps +=1

            if terminated or truncated:
                break

        agent.decay_eps()
        #Update every episode
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(terminated)
        epsilon_history.append(agent.eps)

    env.close()

    results={
        "episode_rewards": np.array(episode_rewards),
        "episode_lengths": np.array(episode_lengths),
        "episode_successes": np.array(episode_successes),
        "epsilon_history": np.array(epsilon_history),
    }

    if track_visitation:
        results["visitation_map"] = visitation_map
    return results

#To smooth out the results

def moving_average(data, window:int=100):
    data=np.asarray(data, dtype=float)
    if len(data)<window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")







        


            

