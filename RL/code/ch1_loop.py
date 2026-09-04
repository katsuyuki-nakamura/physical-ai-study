"""第1章: エージェントと環境のループを手で回す。

学習は一切しない。でたらめな行動を投げて、環境が何を返すかだけを見る。
"""

from reach_env import ReachEnv

env = ReachEnv(goal_conditioned=True, max_steps=10)
env.action_space.seed(0)  # でたらめさも再現できるように種を固定
obs, info = env.reset(seed=0, options={"goal_id": 0})
print(f"obs after reset: {obs}")
print()

total = 0.0
for t in range(10):
    action = env.action_space.sample()  # でたらめな方策
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    print(
        f"t={t}  action [{action[0]:+.2f} {action[1]:+.2f}]"
        f"  reward {reward:+.3f}  total {total:+.3f}"
        f"  terminated={terminated} truncated={truncated}"
    )

print()
print(f"undiscounted return of this episode = {total:+.3f}")
