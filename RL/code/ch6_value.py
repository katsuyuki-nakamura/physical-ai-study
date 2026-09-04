"""第6章: 価値を定義どおりに見積もる。

V(s) = その状態から先に貰えるリターンの期待値。
定義そのままに、たくさん試して平均を取る（モンテカルロ推定）。

ppo_goal.zip があれば、学習済み方策の価値とも比べる。
"""

import numpy as np

from reach_env import ReachEnv

GAMMA = 0.99
N_EPISODES = 300
NAMES = ["左", "右", "上"]


def discounted_return(rewards) -> float:
    g = 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
    return g


def estimate_value(policy, goal_id: int) -> tuple:
    """原点から始めたときのリターンを N 回集めて平均する。"""
    env = ReachEnv(goal_conditioned=True)
    env.action_space.seed(0)
    returns = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=ep, options={"goal_id": goal_id})
        rewards = []
        for _ in range(env.max_steps):
            obs, r, term, trunc, _ = env.step(policy(obs, env))
            rewards.append(r)
            if term or trunc:
                break
        returns.append(discounted_return(rewards))
    xs = np.array(returns)
    return xs.mean(), xs.std() / np.sqrt(len(xs))


def random_policy(obs, env):
    return env.action_space.sample()


print(f"V(原点) のモンテカルロ推定   gamma={GAMMA}, {N_EPISODES}エピソード")
print()
print(f"{'':14} {'でたらめな方策':>16} {'学習済み方策':>16}")
print("-" * 50)

trained = None
try:
    from stable_baselines3 import PPO

    model = PPO.load("ppo_goal")

    def trained_policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    trained = trained_policy
except Exception:
    pass

for goal_id, name in enumerate(NAMES):
    m, se = estimate_value(random_policy, goal_id)
    left = f"{m:+.1f} ± {se:.1f}"
    if trained is None:
        right = "（ppo_goal.zip なし）"
    else:
        tm, tse = estimate_value(trained, goal_id)
        right = f"{tm:+.1f} ± {tse:.1f}"
    print(f"{'z = ' + name:14} {left:>16} {right:>16}")

print()
print("同じ状態でも方策が変われば価値が変わる。")
print("価値は「状態の性質」ではなく「その方策で動いたときの見込み」。")
print()
print("3つの z で値がほぼ同じなのは、3つの的が原点から等距離だから。")
print("目標の位置を非対称にすれば、z ごとに価値が割れる。")
