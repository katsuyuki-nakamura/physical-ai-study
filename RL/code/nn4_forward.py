"""第4章: 順伝播 ── 観測を入れてから行動が出るまで。

学習はまだしない。数の並びが層を通って形を変える様子だけを見る。
"""

import numpy as np

import mlp

p = mlp.init_params(seed=0)

print("パラメータの形と数:")
for k, v in p.items():
    print(f"  {k:3} {str(v.shape):10} {v.size:>5} 個")
print(f"  {'合計':3} {'':10} {sum(v.size for v in p.values()):>5} 個")
print()

# 3つの観測を一度に流す。位置は原点、目標だけが違う。
x = np.zeros((3, 5))
x[0, 2] = x[1, 3] = x[2, 4] = 1.0   # one-hot の z

y, (_, h1, h2) = mlp.forward(p, x)

print("数の並びが形を変えていく:")
print(f"  入力（観測）  {x.shape}   ← 位置2 + 目標3")
print(f"  第1層の後     {h1.shape}   ← ニューロン64本ぶん")
print(f"  第2層の後     {h2.shape}")
print(f"  出力（行動）  {y.shape}   ← 速度指令2つ")
print()

print("まだ学習していないので、出てくる行動はでたらめ:")
for name, row in zip(["left", "right", "up"], y):
    print(f"  z = {name:5} → 行動 {np.round(row, 4)}")
