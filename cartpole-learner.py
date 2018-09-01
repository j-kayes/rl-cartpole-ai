import gym
import random
import numpy as np
import tensorflow as tf
from statistics import mean,  median
from statistics import mean, median
from collections import Counter

lr = 1e-3
env = gym.make('CartPole-v0') 
env.reset()
goal_steps = 200
score_requirement = 50
initial_games = 10000

def random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render() # Display openAI game enviroment
            action = env.action_space.sample() # A random action for the agent to take
            observation, reward, done, info = env.step(action) # Perform the action and record data
            if done:
                break

def initial_population():
    training_data = []
    scores  = []
    accepted_scores = []

    for game in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for step in range(goal_steps):
            action = random.randrange(0,2) # 0 or 1
            observation, reward, done, info = env.step(action) # Perform the action and record data

            if(len(prev_observation) > 0):
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
            
        if(score >= score_requirement):
            accepted_scores.append(score)
            for data in game_memory:
                