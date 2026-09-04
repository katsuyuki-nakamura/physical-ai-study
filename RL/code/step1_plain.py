"""Step 1: 目標ベクトルなしで学習する（＝失敗を見る）。

目標は毎エピソード変わるのに、エージェントには見えない。
すると方策は「3つの的の真ん中あたり」に行くのが最善になる。どの的にも届かない。
"""

import numpy as np
from stable_baselines3 import PPO

from reach_env import ReachEnv, TARGETS, N_GOALS

TOTAL_TIMESTEPS = 150_000


def rollout(model, env, goal_id: int) -> np.ndarray:
    """指定した目標でエピソードを1本走らせ、最終位置を返す。"""
    obs, _ = env.reset(seed=0, options={"goal_id": goal_id})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.pos.copy()


def main():
    env = ReachEnv(goal_conditioned=False)
    print(f"観測の次元: {env.observation_space.shape[0]}  （位置だけ）")

    model = PPO("MlpPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_plain")

    print("\n--- 結果 ---")
    for goal_id in range(N_GOALS):
        final = rollout(model, env, goal_id)
        dist = np.linalg.norm(final - TARGETS[goal_id])
        print(
            f"目標 {goal_id} {TARGETS[goal_id]} → 到達点 [{final[0]:+.2f} {final[1]:+.2f}]"
            f"  距離 {dist:.3f}"
        )
    print("\n到達点が3つとも同じなら、狙い通り「失敗」しています。")


if __name__ == "__main__":
    main()
