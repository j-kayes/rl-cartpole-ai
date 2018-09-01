import gym
import random
import numpy as np
import tensorflow as tf
from statistics import mean, median
from collections import Counter

lr = 1e-3
gamma = 0.95 # Discount rate
env = gym.make('CartPole-v0') 
env.reset()

# Run n games with a randomly generated input, and record the games that just so happen to be above some threshold:
def random_games(n_games = 1000, score_threshold = 50, goal_steps = 200):
    training_data = []
    scores  = []
    accepted_scores = []

    for game in range(n_games):
        score = 0
        game_memory = []
        current_state = []
        for step in range(goal_steps):
            random_var = random.randrange(0,2) # 0 or 1
            action = random_var  # Random action, needs to be 0 or 1 for env.step
            next_state, reward, done, info = env.step(action) # Perform the action and record data
            # Observation/prev_observation contains the 4 variables used to represent the state of the game:
            if(len(current_state) > 0):
                # Append the list with previous state/observation and the current action performed that will lead to the new observation
                action = [1, 0] if random_var else [0, 1] # Update action with array of possibilities
                game_memory.append((current_state, action, reward, next_state, done)) 

            current_state = next_state
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
    
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print('Number of accepted scores:', len(accepted_scores))
    print(Counter(accepted_scores))

    # After all random games, we return the data from those games that met the threshold for training on:
    return training_data

random_games()

def fully_connected(input_tensor, outputs, activation_func=tf.nn.relu):
    result = tf.layers.dense(inputs = input_tensor,
                            units = outputs,
                            activation = activation_func,
                            kernel_initializer = tf.contrib.layers.xavier_initializer())
    return result

# This will be used as the Q*(S,A) estimator, the input is the state and actions performed, and the output is the expected reward:
def neural_network_model(input_size):
    input_data = tf.placeholder(tf.float32, [None, input_size, 1], name = "x")

    fc1 = tf.layers.dropout(fully_connected(input_data, 128), rate=0.8)
    fc2 = tf.layers.dropout(fully_connected(fc1, 128), rate=0.8)
    fc3 = tf.layers.dropout(fully_connected(fc2, 128), rate=0.8)
    fc4 = tf.layers.dropout(fully_connected(fc3, 128), rate=0.8)
    fc5 = tf.layers.dropout(fully_connected(fc4, 128), rate=0.8)

    # Output layer represents the expected reward:
    output_layer = fully_connected(fc5, 1)

    return output_layer

def get_output(session, network, input_data):
    return session.run(network, feed_dict={input_data: input_data})

def get_best_reward(session, network, state_data):
    outputs = []
    for action in env.action_space:
        action = [1, 0] if action else [0, 1]
        x_input = state_data + action # Concatenate the two lists to provide the full S,a input needed.
        outputs.append(get_output(session, network, x_input))
    return max(outputs)

def train_model(session, training_data, batch_size, network, saved_model=False):
    minibatch = random.sample(training_data, batch_size)
    for state, action, reward, next_state, done in training_data:
        if(not done):
            # Reward + discounted best reward for the next state:
            target = (reward + gamma*get_best_reward(session, network, next_state))
        else:
            target = reward
    x_input = state
    cost = tf.squared_difference(target, get_output(session, network, ))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)