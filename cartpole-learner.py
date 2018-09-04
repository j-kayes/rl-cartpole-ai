import gym
import random
import numpy as np
import tensorflow as tf
from agent import *

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0] 

with Agent(env, state_size) as agent:
    #agent.train_network()
    agent.load_model('saved/model.ckpt')
    agent.play(100, epsilo=0.0)

print('Done!')