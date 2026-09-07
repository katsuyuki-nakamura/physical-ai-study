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

print("W shape   :", W.shape, " <- (number of neurons, number of inputs)")
print("x shape   :", x.shape)
print("out shape :", out_matrix.shape, " <- one number per neuron")
print()
print("loop   first 3:", np.round(out_loop[:3], 4))
print("matrix first 3:", np.round(out_matrix[:3], 4))
print("exactly equal :", np.allclose(out_loop, out_matrix))
print()
print(f"parameters in this layer = weights {W.size} (= 64 x 5) + biases {b.size} = {W.size + b.size}")
