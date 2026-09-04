"""2次元リーチング環境。

エージェントは平面上の点。毎エピソード、3つの的のどれか1つが「今回の目標」になる。
goal_conditioned=False なら、観測は自分の位置だけ（＝目標が見えない）。
goal_conditioned=True なら、観測に目標ベクトル z（one-hot）が付く。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 3つの的。左・右・上。
TARGETS = np.array(
    [
        [-0.7, 0.0],  # 0: 左
        [0.7, 0.0],   # 1: 右
        [0.0, 0.7],   # 2: 上
    ],
    dtype=np.float32,
)
N_GOALS = len(TARGETS)


class ReachEnv(gym.Env):
    def __init__(self, goal_conditioned: bool, max_steps: int = 60):
        self.goal_conditioned = goal_conditioned
        self.max_steps = max_steps

        # 行動は2次元の速度指令。[-1, 1] に収める。
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # ここが今回の主役。z を足すと観測が 2 次元から 2 + 3 次元になる。
        obs_dim = 2 + (N_GOALS if goal_conditioned else 0)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        if not self.goal_conditioned:
            return self.pos.copy()
        z = np.zeros(N_GOALS, dtype=np.float32)
        z[self.goal_id] = 1.0  # ← 目標ベクトル
        return np.concatenate([self.pos, z]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.zeros(2, dtype=np.float32)  # 毎回原点から始める
        self.t = 0
        # options で目標を指定できる。指定がなければランダム。
        if options is not None and "goal_id" in options:
            self.goal_id = int(options["goal_id"])
        else:
            self.goal_id = int(self.np_random.integers(N_GOALS))
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.pos = np.clip(self.pos + 0.05 * a, -1.0, 1.0)
        self.t += 1

        dist = float(np.linalg.norm(self.pos - TARGETS[self.goal_id]))
        reward = -dist  # 目標に近いほど高い

        terminated = False
        truncated = self.t >= self.max_steps
        info = {"dist": dist, "goal_id": self.goal_id}
        return self._get_obs(), reward, terminated, truncated, info
