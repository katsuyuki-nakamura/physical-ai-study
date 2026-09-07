"""第7章: 学習させて、動かす。

正解の行動を教師データにして自作 MLP を訓練し、
そのあと自分で60ステップ動かして、3つの的に届くか確かめる。
"""

import numpy as np

import mlp

TARGETS = np.array([[-0.7, 0.0], [0.7, 0.0], [0.0, 0.7]])
NAMES = ["left", "right", "up"]
STEP = 0.05
EPOCHS = 300
BATCH = 256
LR = 0.5


def make_batch(n, rng):
    pos = rng.uniform(-1.0, 1.0, size=(n, 2))
    gid = rng.integers(3, size=n)
    x = np.concatenate([pos, np.eye(3)[gid]], axis=1)
    y = np.clip((TARGETS[gid] - pos) / STEP, -1.0, 1.0)
    return x, y


def loss_and_grad(p, x, y):
    pred, cache = mlp.forward(p, x)
    n = x.shape[0] * mlp.N_OUT
    return float(np.mean((pred - y) ** 2)), mlp.backward(p, cache, 2.0 * (pred - y) / n)


rng = np.random.default_rng(0)
p = mlp.init_params(seed=0)
x_test, y_test = make_batch(2000, np.random.default_rng(999))

print(f"epochs {EPOCHS}  batch {BATCH}  learning rate {LR}")
print()
for epoch in range(EPOCHS + 1):
    x, y = make_batch(BATCH, rng)
    loss, grads = loss_and_grad(p, x, y)
    if epoch % 50 == 0:
        test_loss, _ = loss_and_grad(p, x_test, y_test)
        print(f"  epoch {epoch:>4}   training loss {loss:.5f}   held-out loss {test_loss:.5f}")
    mlp.sgd_step(p, grads, LR)

print()
print("running the trained policy for 60 steps (starting at the origin):")
for gid, name in enumerate(NAMES):
    pos = np.zeros(2)
    z = np.eye(3)[gid]
    for _ in range(60):
        obs = np.concatenate([pos, z])[None]          # (1, 5)
        action, _ = mlp.forward(p, obs)
        pos = np.clip(pos + STEP * np.clip(action[0], -1.0, 1.0), -1.0, 1.0)
    dist = float(np.linalg.norm(pos - TARGETS[gid]))
    print(f"  z = {name:5} target {TARGETS[gid]} -> reached [{pos[0]:+.3f} {pos[1]:+.3f}]  dist {dist:.4f}")

print()
print("One network reads z and picks out a different target each time.")
print("There is a single set of weights; it was not retrained per target.")
