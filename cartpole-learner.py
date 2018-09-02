import gym
import random
import numpy as np
import tensorflow as tf
from statistics import mean, median
from collections import Counter
from agent import *

lr = 1e-3
gamma = 0.95 # Discount rate
env = gym.make('CartPole-v0')
state_len = env.observation_space.shape[0] 

with Agent(env, state_len) as agent:
    print('Getting samples...')
    agent.get_samples(10000)

print('Done!')