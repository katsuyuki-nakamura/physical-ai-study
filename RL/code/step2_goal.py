"""Step 2: 観測に目標ベクトル z を足して学習する。

変えたのは env の引数 1つだけ。アルゴリズムも学習量も Step 1 と同じ。
"""

import numpy as np
from stable_baselines3 import PPO

from reach_env import ReachEnv, TARGETS, N_GOALS

TOTAL_TIMESTEPS = 150_000


def rollout(model, env, goal_id: int) -> np.ndarray:
    obs, _ = env.reset(seed=0, options={"goal_id": goal_id})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.pos.copy()


def main():
    env = ReachEnv(goal_conditioned=True)  # ← 変えたのはここだけ
    print(f"obs dim: {env.observation_space.shape[0]}  (position 2 + goal vector 3)")

    model = PPO("MlpPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_goal")

    print("\n--- result ---")
    for goal_id in range(N_GOALS):
        final = rollout(model, env, goal_id)
        dist = np.linalg.norm(final - TARGETS[goal_id])
        print(
            f"goal {goal_id} {TARGETS[goal_id]} -> reached [{final[0]:+.2f} {final[1]:+.2f}]"
            f"  dist {dist:.3f}"
        )
    print("\nIf the three end up apart, each near its own target, it worked.")
    print("Saved the model to ppo_goal.zip")


if __name__ == "__main__":
    main()
