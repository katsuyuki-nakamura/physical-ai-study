"""第2章: ニューロンを64本並べると、行列の掛け算になる。

1本ずつループで計算しても、行列積で一度に計算しても、答えは同じ。
"""

import numpy as np

rng = np.random.default_rng(0)

x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
W = rng.normal(0.0, 0.5, size=(64, 5))   # 64本 × 入力5個
b = rng.normal(0.0, 0.1, size=64)        # 64本ぶんのバイアス

# 1本ずつ計算する
out_loop = np.zeros(64)
for j in range(64):
    out_loop[j] = np.dot(W[j], x) + b[j]

# 64本まとめて計算する
out_matrix = W @ x + b

print("W の形   :", W.shape, " ← (ニューロンの本数, 入力の数)")
print("x の形   :", x.shape)
print("出力の形 :", out_matrix.shape, " ← ニューロンの本数だけ数が出てくる")
print()
print("ループ版 先頭3個:", np.round(out_loop[:3], 4))
print("行列版   先頭3個:", np.round(out_matrix[:3], 4))
print("完全に一致するか:", np.allclose(out_loop, out_matrix))
print()
print(f"この層のパラメータ数 = 重み {W.size} (= 64 x 5) + バイアス {b.size} = {W.size + b.size}")
