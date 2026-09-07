"""第5章: 損失と勾配 ── 「どちらに動かせば良くなるか」を数で出す。

正解の行動が分かる問題にして、出力とのズレを損失にする。
重みを少しだけ動かして損失の変化を見れば、それが勾配（数値微分）。
"""

import numpy as np

import mlp

TARGETS = np.array([[-0.7, 0.0], [0.7, 0.0], [0.0, 0.7]])
STEP = 0.05   # 1ステップで進める距離


def make_batch(n, rng):
    """観測と、そのときの正解の行動を作る。"""
    pos = rng.uniform(-1.0, 1.0, size=(n, 2))
    gid = rng.integers(3, size=n)
    z = np.eye(3)[gid]                                  # one-hot
    x = np.concatenate([pos, z], axis=1)
    y = np.clip((TARGETS[gid] - pos) / STEP, -1.0, 1.0)  # 目標へ、行き過ぎない分だけ
    return x, y


def loss_of(p, x, y):
    """平均二乗誤差。出力と正解のズレを1つの数にまとめる。"""
    pred, _ = mlp.forward(p, x)
    return float(np.mean((pred - y) ** 2))


rng = np.random.default_rng(0)
x, y = make_batch(256, rng)
p = mlp.init_params(seed=0)

print("observation  :", np.round(x[0], 3))
print("right action :", np.round(y[0], 3))
print()

base = loss_of(p, x, y)
print(f"loss right now: {base:.6f}")
print()

# W1 の左上の重みを1つだけ、少しだけ動かしてみる
eps = 1e-4
p["W1"][0, 0] += eps
plus = loss_of(p, x, y)
p["W1"][0, 0] -= 2 * eps
minus = loss_of(p, x, y)
p["W1"][0, 0] += eps   # 元に戻す

print(f"move W1[0,0] by +{eps}: loss {plus:.6f}")
print(f"move W1[0,0] by -{eps}: loss {minus:.6f}")
print()
print(f"  slope = (up - down) / (2 * {eps}) = {(plus - minus) / (2 * eps):+.6f}")
print()
print("That is the gradient for this one weight. Move it against the sign and the loss drops.")
print("Doing this for every weight would train the net, but there are far too many.")
print(f"(this net has {sum(v.size for v in p.values())} weights. One update would need {sum(v.size for v in p.values()) * 2} forward passes)")
