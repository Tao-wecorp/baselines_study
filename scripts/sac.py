#! /usr/bin/env python

import gym

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines import A2C, SAC, PPO2, TD3

import utils.warning_ignore


env = gym.make('Pendulum-v0')
env = DummyVecEnv([lambda: env])

model = SAC('MlpPolicy', 'Pendulum-v0', batch_size=256, verbose=1, policy_kwargs=dict(layers=[256, 256]), seed=0)
model.learn(total_timesteps=8000, log_interval=10)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")