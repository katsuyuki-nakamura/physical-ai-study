"""第2章: その観測だけで行動を決められるか（マルコフ性）を確かめる。

同じ状況で観測が一致してしまうなら、観測だけでは正しい行動を選べない。
"""

import numpy as np

from reach_env import ReachEnv

for goal_conditioned in (False, True):
    env = ReachEnv(goal_conditioned=goal_conditioned)
    print(f"--- goal_conditioned={goal_conditioned} ---")

    seen = []
    for goal_id, name in enumerate(["左", "右", "上"]):
        obs, _ = env.reset(seed=0, options={"goal_id": goal_id})
        seen.append(obs.copy())
        print(f"  目標={name}  reset 直後の観測: {obs}")

    same = all(np.array_equal(seen[0], o) for o in seen)
    print(f"  3つとも同じ観測か: {same}")
    print("  →", "区別できない（マルコフでない）" if same else "区別できる（マルコフ）")
    print()
