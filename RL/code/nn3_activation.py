"""第3章: 活性化関数を挟まないと、層を重ねる意味が消える。

線形な層をいくつ重ねても、1つの行列で書けてしまう。
"""

import numpy as np

rng = np.random.default_rng(0)

x = rng.normal(size=5)
W1 = rng.normal(0.0, 0.5, size=(8, 5))
W2 = rng.normal(0.0, 0.5, size=(4, 8))

two_layers = W2 @ (W1 @ x)   # 2層ぶん計算する
one_layer = (W2 @ W1) @ x    # 重みを先に掛けて1層にまとめる

print("--- 活性化なし ---")
print("  2層で計算 :", np.round(two_layers, 4))
print("  1層にまとめ:", np.round(one_layer, 4))
print("  一致するか :", np.allclose(two_layers, one_layer))
print("  → 何層重ねても1層で書ける。深くした意味がない。")
print()

with_tanh = W2 @ np.tanh(W1 @ x)   # 途中に tanh を挟む

print("--- tanh あり ---")
print("  2層で計算 :", np.round(with_tanh, 4))
print("  1層で書けるか:", np.allclose(with_tanh, one_layer))
print("  → 書けない。非線形が入って初めて、層を増やす意味が出る。")
print()

print("tanh は値を -1〜1 に押し込む関数:")
for v in (-3.0, -1.0, 0.0, 1.0, 3.0):
    print(f"  tanh({v:+.1f}) = {np.tanh(v):+.4f}")
