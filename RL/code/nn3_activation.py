"""第3章: 活性化関数を挟まないと、層を重ねる意味が消える。

線形な層をいくつ重ねても、1つの行列で書けてしまう。
"""

import numpy as np

rng = np.random.default_rng(0)

x = rng.normal(size=5)
W1 = rng.normal(0.0, 0.5, size=(64, 5))
W2 = rng.normal(0.0, 0.5, size=(2, 64))

two_layers = W2 @ (W1 @ x)   # 2層ぶん計算する
one_layer = (W2 @ W1) @ x    # 重みを先に掛けて1層にまとめる

print("--- without activation ---")
print("  two layers    :", np.round(two_layers, 4))
print("  folded to one :", np.round(one_layer, 4))
print("  equal         :", np.allclose(two_layers, one_layer))
print("  -> any number of layers collapses into one. Depth buys nothing.")
print()

with_tanh = W2 @ np.tanh(W1 @ x)   # 途中に tanh を挟む

print("--- with tanh ---")
print("  two layers    :", np.round(with_tanh, 4))
print("  same as one   :", np.allclose(with_tanh, one_layer))
print("  -> no. Depth starts to mean something only once a nonlinearity sits in between.")
print()

print("tanh squeezes any value into -1..1:")
for v in (-3.0, -1.0, 0.0, 1.0, 3.0):
    print(f"  tanh({v:+.1f}) = {np.tanh(v):+.4f}")
