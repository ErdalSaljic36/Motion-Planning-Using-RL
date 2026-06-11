from maze_env_window import MazeEnvWindow
import time
import numpy as np


env = MazeEnvWindow(render_mode ="human")
obs, info = env.reset()


print ("Observation space: ", env.observation_space)
print("Initial observation: ")
print(obs)

env.render()
time.sleep(1.5)

print("Random walk:")

for i in range(30):
    action = env.action_space.sample()  #Pick a random action from action space
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.25)
    if terminated or truncated:
        break
print("\nLast observation: ")
print(obs)
env.close()
    
