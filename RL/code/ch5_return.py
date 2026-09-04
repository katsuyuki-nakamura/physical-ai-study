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


print(f"{'gamma':>6} | {'いますぐ+1':>12} | {'9手後に+10':>12} | 選ぶのは")
print("-" * 50)
for gamma in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0):
    a = discounted_return(NOW, gamma)
    b = discounted_return(LATER, gamma)
    print(f"{gamma:>6.2f} | {a:>12.4f} | {b:>12.4f} | {'いますぐ' if a > b else 'あとで'}")

print()
print("gamma は「どれだけ先まで気にするか」を決めるつまみ。")
print("0 なら次の一手だけ、1 に近いほど遠い将来まで勘定に入れる。")
