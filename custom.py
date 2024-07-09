import gym
import json
import datetime as dt

# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
# import talib as ta
from env.StockTradingEnv import StockTradingEnv

import pandas as pd

df = pd.read_csv('./data/AAPL.csv')
df = df.sort_values('Date')
df.dropna(inplace=True)
df = df.sort_values('Date')
df = df.reset_index()

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render('human')
env.close()