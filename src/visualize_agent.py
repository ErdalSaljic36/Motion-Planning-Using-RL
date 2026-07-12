import pickle
import time
import numpy as np

from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO
from maze_env_goal import MazeEnvGoal


AGENT = "LSTM"          # "DQN" or "LSTM"
WINDOW_SIZE = 3
STEP_DELAY = 0.15      

MODEL_PATHS = {
    "DQN": "../models/dqn_generalize_w3_5kmazes",
    "LSTM": "../models/lstm_ppo_w3_5000mazes",
}


with open("../datasets/test_mazes500.pkl", "rb") as f:
    test_mazes = pickle.load(f)

print(f"Loaded {len(test_mazes)} test mazes")

# Pick three mazes: short, medium, long optimal path
# Sort by optimal path length
sorted_mazes = sorted(test_mazes, key=lambda m: m["optimal_len"])

lengths = [m["optimal_len"] for m in sorted_mazes]
print(f"Optimal path lengths in test set: "
      f"min={min(lengths)}, max={max(lengths)}")

# Short: near the bottom, Medium: middle, Long: near the top
n = len(sorted_mazes)
selected = [
    ("SHORT ", sorted_mazes[int(n * 0.80)]),   
    ("MEDIUM", sorted_mazes[int(n * 0.89)]),   
    ("LONG  ", sorted_mazes[int(n * 0.938)]),   
]

print("\nSelected mazes:")
for label, m in selected:
    print(f"  {label}: optimal path = {m['optimal_len']} steps")


if AGENT == "DQN":
    model = DQN.load(MODEL_PATHS["DQN"])
    is_recurrent = False
else:
    model = RecurrentPPO.load(MODEL_PATHS["LSTM"])
    is_recurrent = True

print(f"\nLoaded {AGENT} model\n")

print("=" * 60)
print(f"{AGENT} on three test mazes (unseen during training)")
print("=" * 60)

results = []

for label, maze_data in selected:
    env = MazeEnvGoal(window_size=WINDOW_SIZE, maze_set=[maze_data],
                      render_mode="human")
    obs, info = env.reset()

    # LSTM hidden state must be carried through the episode
    lstm_state = None
    episode_start = True

    terminated = truncated = False
    steps = 0

    print(f"\n{label} maze  |  optimal = {maze_data['optimal_len']} steps")

    env.render()
    time.sleep(1.0)   # pause so the starting layout is visible

    while not (terminated or truncated):
        if is_recurrent:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=np.array([episode_start]),
                deterministic=True,
            )
            episode_start = False
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        steps += 1

        env.render()
        time.sleep(STEP_DELAY)

    if terminated:
        ratio = steps / maze_data["optimal_len"]
        print(f"  SUCCESS in {steps} steps  (ratio {ratio:.2f}x optimal)")
        results.append((label, True, steps, ratio))
    else:
        print(f"  FAILED  (gave up after {steps} steps)")
        results.append((label, False, steps, None))

    time.sleep(0.8)
    env.close()


print("\n" + "=" * 60)
print(f"SUMMARY: {AGENT}")
print("=" * 60)
for label, success, steps, ratio in results:
    status = "SUCCESS" if success else "FAILED "
    ratio_str = f"{ratio:.2f}x" if ratio else "---"
    print(f"{label}  {status}  steps={steps:<5} ratio={ratio_str}")
print("=" * 60)