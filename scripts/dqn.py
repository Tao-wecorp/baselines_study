# 
#! /usr/bin/env python

import gym

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines import DQN

import utils.warning_ignore


env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)}
model = DQN('MlpPolicy', env, verbose=1, **kwargs)
model.learn(total_timesteps=10000, log_interval=10)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")