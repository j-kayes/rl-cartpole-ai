import gym
import random
import numpy as np
import tensorflow as tf
from statistics import mean, median
from collections import Counter

lr = 1e-3
env = gym.make('CartPole-v0') 
env.reset()

# Run n games with a randoly generated input, and record the games that just so happen to be above some threshold:
def random_games(n_games = 1000, score_threshold = 50, goal_steps = 200):
    training_data = []
    scores  = []
    accepted_scores = []

    for game in range(n_games):
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
        
        scores.append(score)
        # Successful games get added to the training data:
        if(score >= score_threshold):
            accepted_scores.append(score)
            for step_data in game_memory:
                training_data.append(step_data)
        env.reset()
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    # After all random games, we return the data from those games that met the threshold for training on:
    return training_data

random_games()