import gym
import random
import numpy as np
import tensorflow as tf
from agent import *

lr = 5e-5
env = gym.make('CartPole-v0')
state_len = env.observation_space.shape[0] 

with Agent(env, state_len, lr) as agent:
    agent.train_network()

print('Done!')