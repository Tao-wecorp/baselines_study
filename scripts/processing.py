#! /usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt
import gym

from stable_baselines.bench import Monitor
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

import utils.warning_ignore


env_id = "CartPole-v1"
num_cpu = 4
ALGO = PPO2

env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
model = ALGO('MlpPolicy', env, verbose=0)
start = time.time()
model.learn(total_timesteps=1000)
print(time.time()-start)
env.close()

eval_env = gym.make(env_id)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')