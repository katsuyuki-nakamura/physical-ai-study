"""numpy だけで書いた2層の多層パーセプトロン（MLP）。

観測5個を受け取り、行動2個を返す。5 → 64 → 64 → 2。
学習に要る3つ（順伝播・逆伝播・更新）をこの1ファイルに入れてある。
"""

import numpy as np

N_IN, N_HIDDEN, N_OUT = 5, 64, 2


def init_params(seed: int = 0) -> dict:
    """重みを乱数で、バイアスをゼロで初期化する。"""
    rng = np.random.default_rng(seed)

    def w(n_in, n_out):
        # 入力が多いほど小さく振る（He / Xavier 初期化の考え方）
        return rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_out, n_in))

    return {
        "W1": w(N_IN, N_HIDDEN), "b1": np.zeros(N_HIDDEN),
        "W2": w(N_HIDDEN, N_HIDDEN), "b2": np.zeros(N_HIDDEN),
        "W3": w(N_HIDDEN, N_OUT), "b3": np.zeros(N_OUT),
    }


def forward(p: dict, x: np.ndarray):
    """x: (N, 5) を入れて行動 (N, 2) を返す。逆伝播で使う途中の値も返す。"""
    h1 = np.tanh(x @ p["W1"].T + p["b1"])
    h2 = np.tanh(h1 @ p["W2"].T + p["b2"])
    y = h2 @ p["W3"].T + p["b3"]
    return y, (x, h1, h2)


def backward(p: dict, cache, dy: np.ndarray) -> dict:
    """出力側の勾配 dy から、各パラメータの勾配を求める（連鎖律）。"""
    x, h1, h2 = cache
    g = {"W3": dy.T @ h2, "b3": dy.sum(axis=0)}

    d2 = (dy @ p["W3"]) * (1.0 - h2 ** 2)      # tanh の微分は 1 - tanh^2
    g["W2"], g["b2"] = d2.T @ h1, d2.sum(axis=0)

    d1 = (d2 @ p["W2"]) * (1.0 - h1 ** 2)
    g["W1"], g["b1"] = d1.T @ x, d1.sum(axis=0)
    return g


def sgd_step(p: dict, g: dict, lr: float) -> None:
    """勾配の逆向きにパラメータを少しだけ動かす。"""
    for k in p:
        p[k] -= lr * g[k]
