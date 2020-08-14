#! /usr/bin/env python

import gym

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import utils.warning_ignore


class CustomPolicy(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                        net_arch=[8, 'lstm', dict(pi=[2048, 1024, 512, 256, 128], vf=[2048, 1024, 512, 256, 128])],
                                        layer_norm=True, feature_extraction="mlp")


env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = PPO2(CustomPolicy, env, nminibatches=1, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()