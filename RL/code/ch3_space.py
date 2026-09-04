"""第3章: 行動空間と観測空間の中身を覗く。"""

import numpy as np

from reach_env import ReachEnv

env = ReachEnv(goal_conditioned=True)
env.action_space.seed(0)

print("action_space      :", env.action_space)
print("  low  / high     :", env.action_space.low, "/", env.action_space.high)
print("  sample()        :", env.action_space.sample())
print()
print("observation_space :", env.observation_space)
print("  shape           :", env.observation_space.shape)
print()

print("宣言した範囲の外を投げるとどうなるか")
for a in ([10.0, 0.0], [1.0, 0.0]):
    env.reset(seed=0, options={"goal_id": 1})
    env.step(np.array(a, dtype=np.float32))
    print(f"  行動 {a} → 位置 {env.pos}")
print("→ step() の中で clip しているので同じ結果になる。宣言は約束であって、強制ではない。")
