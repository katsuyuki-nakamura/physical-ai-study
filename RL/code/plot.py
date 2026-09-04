"""おまけ: 2つの方策の軌跡を並べて描く。step1 と step2 を実行した後に走らせる。

図中のラベルは英語にしてある。matplotlib は既定で日本語フォントを持たず、
日本語を書くと豆腐（□）になるため。フォント設定に寄り道しないための割り切り。
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from reach_env import ReachEnv, TARGETS, N_GOALS

COLORS = ["#c0392b", "#2471a3", "#1e8449"]
LABELS = ["left", "right", "up"]


def trajectory(model, goal_conditioned: bool, goal_id: int) -> np.ndarray:
    env = ReachEnv(goal_conditioned=goal_conditioned)
    obs, _ = env.reset(seed=0, options={"goal_id": goal_id})
    path = [env.pos.copy()]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        path.append(env.pos.copy())
        done = terminated or truncated
    return np.array(path)


def draw(ax, model, goal_conditioned: bool, title: str):
    prefix = "z = " if goal_conditioned else "goal: "
    for goal_id in range(N_GOALS):
        path = trajectory(model, goal_conditioned, goal_id)
        # 太さを変えて描く。Step 1 では3本が完全に重なるので、
        # そのままだと最後の1本しか見えない。
        ax.plot(path[:, 0], path[:, 1], color=COLORS[goal_id], lw=6 - 2 * goal_id,
                alpha=0.9, label=f"{prefix}{LABELS[goal_id]}")
        ax.plot(path[-1, 0], path[-1, 1], "o", color=COLORS[goal_id], ms=7)
    for goal_id in range(N_GOALS):
        ax.plot(*TARGETS[goal_id], "x", color=COLORS[goal_id], ms=13, mew=3)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.4, 1.0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    draw(axes[0], PPO.load("ppo_plain"), False,
         "Step 1: without z  (same spot for every goal)")
    draw(axes[1], PPO.load("ppo_goal"), True,
         "Step 2: with z  (one policy, three targets)")
    fig.tight_layout()
    fig.savefig("trajectories.png", dpi=140)
    print("wrote trajectories.png")


if __name__ == "__main__":
    main()
