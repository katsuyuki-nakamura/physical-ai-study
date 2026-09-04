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

print("打ち切った1ステップの学習目標")
print(f"  truncated として扱う（正しい）: {correct:+8.3f}")
print(f"  terminated として扱う（誤り） : {wrong:+8.3f}")
print(f"  ずれ                          : {correct - wrong:+8.3f}")
print()
print("誤ると「時間切れの直前はやけに良い状態だ」と学習してしまう。")
print("max_steps による打ち切りは truncated であって terminated ではない。")
print("だから step() は2つを別の値として返す。")
