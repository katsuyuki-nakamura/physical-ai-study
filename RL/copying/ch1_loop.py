
from reach_env import ReachEnv

env = ReachEnv(goal_condition=True, max_steps=10)
env.action_space.seed(0)
obs, info = env.reset(seed=0, options={"goal_id": 0})
print(f"resetが返した観測: {obs}")
print()

total = 0.0
for t in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    print(
        f"t={t} 行動 [{action[0]:+.2f} {action[1]:+.2f}]"
        f"  報酬 {reward:+0.3f} 累計 {total:+.3f}"
        f"  terminated={terminated} truncated={truncated}"
    )

print()
print(f"このエピソードのリターン（割引なし）={total:+.3f}")
