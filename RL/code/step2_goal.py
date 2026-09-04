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
    print(f"観測の次元: {env.observation_space.shape[0]}  （位置2 + 目標ベクトル3）")

    model = PPO("MlpPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_goal")

    print("\n--- 結果 ---")
    for goal_id in range(N_GOALS):
        final = rollout(model, env, goal_id)
        dist = np.linalg.norm(final - TARGETS[goal_id])
        print(
            f"目標 {goal_id} {TARGETS[goal_id]} → 到達点 [{final[0]:+.2f} {final[1]:+.2f}]"
            f"  距離 {dist:.3f}"
        )
    print("\n到達点が3つとも違い、それぞれの的に寄っていれば成功です。")
    print("モデルを ppo_goal.zip に保存しました。")


if __name__ == "__main__":
    main()
