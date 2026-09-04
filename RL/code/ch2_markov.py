"""第2章: その観測だけで行動を決められるか（マルコフ性）を確かめる。

同じ状況で観測が一致してしまうなら、観測だけでは正しい行動を選べない。
"""

import numpy as np

from reach_env import ReachEnv

for goal_conditioned in (False, True):
    env = ReachEnv(goal_conditioned=goal_conditioned)
    print(f"--- goal_conditioned={goal_conditioned} ---")

    seen = []
    for goal_id, name in enumerate(["left", "right", "up"]):
        obs, _ = env.reset(seed=0, options={"goal_id": goal_id})
        seen.append(obs.copy())
        print(f"  goal={name:<5} obs after reset: {obs}")

    same = all(np.array_equal(seen[0], o) for o in seen)
    print(f"  all three identical: {same}")
    print("  ->", "indistinguishable (not Markov)" if same else "distinguishable (Markov)")
    print()
