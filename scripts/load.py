#! /usr/bin/env python

import gym

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines import A2C, SAC, PPO2, TD3

import utils.warning_ignore


checkpoints_dir = "checkpoint" 
env = gym.make('Pendulum-v0')
env = DummyVecEnv([lambda: env])

loaded_model = PPO2.load(checkpoints_dir + "/PPO2_Pendulum")
# print("loaded:", "gamma =", loaded_model.gamma, "n_steps =", loaded_model.n_steps)

mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

loaded_model.set_env(DummyVecEnv([lambda: gym.make('Pendulum-v0')]))
loaded_model.learn(8000)