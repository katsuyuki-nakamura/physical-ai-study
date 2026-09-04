import numpy as np
import gymnasium as gym
import gymnasium import spaces

TARGETS = np.array(
    [
        [-0.7, 0.0], # 0: 左
        [0.7, 0.0],  # 1: 右
        [0,0, 0.7],  # 2: 上
    ],
    dtype=np.float32,
)
N_GOALS = len(TARGETS)

class ReachEnv(gym.Env):
    def __init__(self, goal_conditioned: bool, max_steps: int = 60):
        self.goal_conditioned = goal_conditioned
        self.max_steps = max_steps

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        obs_dim = 2 + (N_GOALS if goal_conditioned else 0)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

    
