#! /usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt
import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import set_random_seed
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3 import PPO, A2C


env_id = "CartPole-v1"
num_cpu = 36
ALGO = A2C

env = make_vec_env(env_id, n_envs=num_cpu)
model = ALGO('MlpPolicy', env, verbose=1)
start = time.time()
model.learn(total_timesteps=100)
print(time.time()-start)
env.close()

eval_env = gym.make(env_id)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')