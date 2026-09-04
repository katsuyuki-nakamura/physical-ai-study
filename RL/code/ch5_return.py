"""第5章: リターンと割引率 gamma。

同じ報酬の並びでも、gamma を変えると「どちらが得か」の判断が変わる。
"""

NOW = [1.0] + [0.0] * 9        # いますぐ +1
LATER = [0.0] * 9 + [10.0]     # 9手待って +10


def discounted_return(rewards, gamma: float) -> float:
    """G = r0 + γr1 + γ²r2 + ... を後ろから畳んで計算する。"""
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


print(f"{'gamma':>6} | {'+1 now':>12} | {'+10 in 9 steps':>14} | pick")
print("-" * 48)
for gamma in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0):
    a = discounted_return(NOW, gamma)
    b = discounted_return(LATER, gamma)
    print(f"{gamma:>6.2f} | {a:>12.4f} | {b:>14.4f} | {'now' if a > b else 'later'}")

print()
print("gamma is the knob for how far ahead you care.")
print("0 looks only at the next step; closer to 1 counts the distant future.")
