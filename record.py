#! /usr/bin/env python

import gym

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines import PPO2

import utils.warning_ignore

def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step==0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    eval_env.close()

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

record_video('CartPole-v1', model, video_length=500, prefix='ppo2-cartpole')