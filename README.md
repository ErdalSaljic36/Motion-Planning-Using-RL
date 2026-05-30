# Motion Planning using Reinforcement Learning

Bachelor thesis project: a point robot learns to navigate through a maze
using Reinforcement Learning, observing only its local surroundings (POMDP).

## Tech stack

- Python 3.12
- NumPy, Gymnasium, Pygame, Matplotlib
- (Later) PyTorch, Stable-Baselines3

## Setup

```bash
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project phases

1. Tabular Q-Learning, full observability
2. Partial observability (local window)
3. DQN with neural networks
4. LIDAR / raycasting observations
5. Dynamic obstacles and generalization
6. Recurrent PPO (optional)

## Author

Erdal Saljic