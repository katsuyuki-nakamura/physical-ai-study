"""第1章: ニューロン1本を手で計算する。

入力に重みを掛けて全部足し、バイアスを足す。ニューロンはそれだけ。
"""

import numpy as np

# 観測（原点にいて、今回の目標は「左」）。強化学習の本と同じ形。
x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

# ニューロン1本ぶんの重みとバイアス。値はいまは適当でよい。
w = np.array([0.5, -0.2, 1.3, 0.1, -0.7])
b = 0.4

print("入力     x =", x)
print("重み     w =", w)
print("バイアス b =", b)
print()

print("1項ずつ書き下すと:")
total = 0.0
for i in range(len(x)):
    term = w[i] * x[i]
    total += term
    print(f"  w[{i}] * x[{i}] = {w[i]:+.1f} * {x[i]:+.1f} = {term:+.2f}")
print(f"  バイアス                    = {b:+.2f}")
total += b
print(f"  合計                        = {total:+.2f}")
print()

print("numpy なら1行:")
print("  np.dot(w, x) + b =", np.dot(w, x) + b)
print()
print("重みは入力と同じ数だけ要る。バイアスはニューロン1本につき1つ。")
