#! /usr/bin/env python

import gym

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, SAC, PPO2, TD3

import utils.warning_ignore


checkpoints_dir = "checkpoint" 
model = PPO2('MlpPolicy', 'Pendulum-v0', verbose=0).learn(8000)

obs = model.env.observation_space.sample()
print("pre saved", model.predict(obs, deterministic=True))

model.save(checkpoints_dir + "/PPO2_Pendulum")