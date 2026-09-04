"""第4章: 密な報酬とまばらな報酬で、返ってくる手がかりの量を比べる。

でたらめな方策で100エピソード回し、報酬がどれだけ情報を持つかを見る。
"""

import numpy as np

from reach_env import ReachEnv

N_EPISODES = 100


def run(kind: str) -> np.ndarray:
    env = ReachEnv(goal_conditioned=True)
    env.action_space.seed(0)
    totals = []
    for ep in range(N_EPISODES):
        env.reset(seed=ep)
        total = 0.0
        for _ in range(env.max_steps):
            _, r, term, trunc, info = env.step(env.action_space.sample())
            if kind == "dense":
                total += r  # -距離 をそのまま
            else:
                total += 1.0 if info["dist"] < 0.1 else 0.0  # 触れたら +1
            if term or trunc:
                break
        totals.append(total)
    return np.array(totals)


dense = run("dense")
sparse = run("sparse")

print(f"{'':10} {'平均':>10} {'標準偏差':>10} {'ゼロでない回':>14}")
print("-" * 48)
print(f"{'密な報酬':10} {dense.mean():>10.2f} {dense.std():>10.2f}"
      f" {np.count_nonzero(dense):>10} /{N_EPISODES}")
print(f"{'疎な報酬':10} {sparse.mean():>10.2f} {sparse.std():>10.2f}"
      f" {np.count_nonzero(sparse):>10} /{N_EPISODES}")
print()
print("疎な報酬では、でたらめに動いている限りほとんど何も返ってこない。")
print("学習の前に「そもそも手がかりがあるか」を確かめるのが報酬設計。")
