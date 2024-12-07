import gym
import numpy as np

import torch

class BaseEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.action_space = gym.spaces.Discrete(config.action_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config.obs_dim,), dtype=np.float32)

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed=None):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_done(self):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError