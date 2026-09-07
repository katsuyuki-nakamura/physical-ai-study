"""第4章: 順伝播 ── 観測を入れてから行動が出るまで。

学習はまだしない。数の並びが層を通って形を変える様子だけを見る。
"""

import numpy as np

import mlp

p = mlp.init_params(seed=0)

print("parameter shapes and counts:")
for k, v in p.items():
    print(f"  {k:5} {str(v.shape):10} {v.size:>5}")
print(f"  {'total':5} {'':10} {sum(v.size for v in p.values()):>5}")
print()

# 3つの観測を一度に流す。位置は原点、目標だけが違う。
x = np.zeros((3, 5))
x[0, 2] = x[1, 3] = x[2, 4] = 1.0   # one-hot の z

y, (_, h1, h2) = mlp.forward(p, x)

print("the numbers change shape on the way through:")
print(f"  input (obs)     {x.shape}   <- position 2 + goal 3")
print(f"  after layer 1   {h1.shape}   <- 64 neurons")
print(f"  after layer 2   {h2.shape}")
print(f"  output (action) {y.shape}   <- 2 velocity commands")
print()

print("not trained yet, so the actions are nonsense:")
for name, row in zip(["left", "right", "up"], y):
    print(f"  z = {name:5} -> action {np.round(row, 4)}")
