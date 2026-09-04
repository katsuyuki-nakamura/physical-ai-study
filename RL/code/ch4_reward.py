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

print(f"{'':8} {'mean':>10} {'std':>10} {'nonzero':>12}")
print("-" * 42)
print(f"{'dense':8} {dense.mean():>10.2f} {dense.std():>10.2f}"
      f" {np.count_nonzero(dense):>8} /{N_EPISODES}")
print(f"{'sparse':8} {sparse.mean():>10.2f} {sparse.std():>10.2f}"
      f" {np.count_nonzero(sparse):>8} /{N_EPISODES}")
print()
print("With a sparse reward, random moves get almost nothing back.")
print("Reward design is about checking there is a signal at all, before training.")
