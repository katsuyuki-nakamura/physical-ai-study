"""第6章: 逆伝播 ── 全部の勾配を、順伝播1回ぶんの手間で出す。

答え合わせは数値微分。第5章のやり方で出した傾きと、
mlp.backward() が返す傾きが一致すれば、逆伝播が正しく書けている。
"""

import numpy as np

import mlp


def loss_and_grad(p, x, y):
    """損失と、全パラメータの勾配を返す。"""
    pred, cache = mlp.forward(p, x)
    n = x.shape[0] * mlp.N_OUT
    loss = float(np.mean((pred - y) ** 2))
    dy = 2.0 * (pred - y) / n          # 損失を出力で微分したもの
    return loss, mlp.backward(p, cache, dy)


def numeric_grad(p, x, y, key, idx, eps=1e-5):
    """パラメータを1つだけ動かして、損失の傾きを測る（第5章と同じ手口）。"""

    def loss_at(delta):
        p[key][idx] += delta
        pred, _ = mlp.forward(p, x)
        p[key][idx] -= delta
        return float(np.mean((pred - y) ** 2))

    return (loss_at(eps) - loss_at(-eps)) / (2 * eps)


rng = np.random.default_rng(0)
x = rng.normal(size=(16, mlp.N_IN))
y = rng.normal(size=(16, mlp.N_OUT))
p = mlp.init_params(seed=0)

loss, grads = loss_and_grad(p, x, y)
print(f"損失: {loss:.6f}")
print()
print(f"{'パラメータ':10} {'逆伝播':>14} {'数値微分':>14} {'差':>12}")
print("-" * 54)

worst = 0.0
for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
    # 各パラメータから2箇所ずつ、重複しないように抜き出して調べる
    for flat in rng.choice(p[key].size, size=2, replace=False):
        idx = np.unravel_index(flat, p[key].shape)
        a = grads[key][idx]
        n = numeric_grad(p, x, y, key, idx)
        worst = max(worst, abs(a - n))
        label = f"{key}[{','.join(map(str, idx))}]"
        print(f"{label:10} {a:>14.9f} {n:>14.9f} {abs(a - n):>12.2e}")

print("-" * 54)
print(f"いちばん大きい差: {worst:.2e}")
print("→ 1e-6 より小さければ、逆伝播は正しく書けている。")
print()
print("数値微分は重み1つにつき順伝播2回。逆伝播は全部まとめて1回。")
print(f"このネットなら {sum(v.size for v in p.values()) * 2} 回 → 1 回。これが逆伝播の値打ち。")
