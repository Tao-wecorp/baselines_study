#! /usr/bin/env python

import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, SAC, PPO, TD3

checkpoints_dir = "checkpoint"
log_dir = "logs"

env_id = 'Pendulum-v0'
env = gym.make(env_id)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])


callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=5000, callback=eval_callback, reset_num_timesteps=False)

model.save(checkpoints_dir + "/PPO2_Pendulum")

obs = model.env.observation_space.sample()
print(model.predict(obs, deterministic=True))