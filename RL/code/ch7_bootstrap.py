"""第7章: terminated と truncated を取り違えると何がずれるか。

価値の更新では「いま貰った報酬 + この先の見込み」を目標にする。
本当に終わった(terminated)なら、この先はゼロ。
時間切れ(truncated)なら、この先はまだ続いている。ゼロにしてはいけない。
"""

GAMMA = 0.99
REWARD = -0.5      # 打ち切り時点で貰った報酬
V_NEXT = -50.0     # 打ち切った先の状態の価値（この先も -0.5 が続く見込み）

correct = REWARD + GAMMA * V_NEXT   # truncated: 先を足す
wrong = REWARD + GAMMA * 0.0        # terminated 扱い: 先をゼロにする

print("learning target for the step that got cut off")
print(f"  treated as truncated (correct): {correct:+8.3f}")
print(f"  treated as terminated (wrong) : {wrong:+8.3f}")
print(f"  gap                           : {correct - wrong:+8.3f}")
print()
print("Get it wrong and the agent learns that just before the time limit is a great place to be.")
print("A cut-off by max_steps is truncated, not terminated.")
print("That is why step() returns the two as separate values.")
