"""Step 3: 学習済みの重みは一切さわらず、z だけを差し替える。

デモは2つ。
  A. 同じ位置で z だけ変えると、出てくる行動が変わる
  B. エピソードの途中で z を書き換えると、進路がその場で変わる（＝介入点）
"""

import numpy as np
from stable_baselines3 import PPO

from reach_env import ReachEnv, TARGETS, N_GOALS

LABELS = ["left", "right", "up"]


def make_obs(pos: np.ndarray, goal_id: int) -> np.ndarray:
    """位置と目標IDから観測を組み立てる。z は one-hot。"""
    z = np.zeros(N_GOALS, dtype=np.float32)
    z[goal_id] = 1.0
    return np.concatenate([pos, z]).astype(np.float32)


def demo_a(model):
    print("=== A. same position, different z ===")
    pos = np.zeros(2, dtype=np.float32)  # 原点に立たせる
    for goal_id in range(N_GOALS):
        obs = make_obs(pos, goal_id)
        action, _ = model.predict(obs, deterministic=True)
        print(
            f"z = {LABELS[goal_id]:<5}({make_obs(pos, goal_id)[2:].astype(int)})"
            f" -> action [{action[0]:+.2f} {action[1]:+.2f}]"
        )
    print("Only the three z entries changed. Same position, same weights.\n")


def demo_b(model, switch_at: int = 20, total: int = 60):
    print(f"=== B. swap z from left to right at step {switch_at} ===")
    env = ReachEnv(goal_conditioned=True, max_steps=total)
    obs, _ = env.reset(seed=0, options={"goal_id": 0})  # 左を目指して出発

    for t in range(total):
        if t == switch_at:
            env.goal_id = 1  # ← 介入。重みは触っていない
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if t % 10 == 0 or t == switch_at:
            mark = "  <- z swapped here" if t == switch_at else ""
            print(f"  t={t:2d}  x={env.pos[0]:+.2f}  y={env.pos[1]:+.2f}{mark}")
        if terminated or truncated:
            break

    print(f"  end   x={env.pos[0]:+.2f}  y={env.pos[1]:+.2f}")
    print("If x first moves negative and then turns back positive, it worked.")


def main():
    model = PPO.load("ppo_goal")
    demo_a(model)
    demo_b(model)


if __name__ == "__main__":
    main()
