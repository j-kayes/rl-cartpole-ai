# Copywrite James Kayes (c) 2018
import tensorflow as tf
import numpy as np
import random
from statistics import mean, median

def fully_connected(input_tensor, outputs, activation_func=tf.nn.relu):
    result = tf.layers.dense(inputs = input_tensor,
                            units = outputs,
                            activation = activation_func,
                            kernel_initializer = tf.contrib.layers.xavier_initializer())
    return result

class Agent: 
    
    def __init__(self, environment, state_size, learning_rate=1e-4, sequence_length=10, memory_size=100000):
        self.env = environment
        self.sequence_length = sequence_length
        self.state_size = state_size

        self.action_space = np.zeros(shape=(self.env.action_space.n,self.env.action_space.n))
        n = 0
        for n in range(self.env.action_space.n):
            self.action_space[n][n] = 1

        # Input size will be the size of the previous sequence/action buffer:
        self.input_size = sequence_length*(self.env.action_space.n + self.state_size) - self.env.action_space.n

        self.memory = []
        self.memory_size = memory_size
        self.memory_index = 0
        
        self.lr = learning_rate
        
        self.build_model()
        # For saving and loading the graph after training:
        self.saver = tf.train.Saver(save_relative_paths=True)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def build_model(self, fc_layers=4, layer_connections=128, hold_rate=1.0):
        # This will be used as the Q*(S,A) estimator:
        self.input_data = tf.placeholder(tf.float32, [None, self.input_size], name = "x")
        self.target = tf.placeholder(tf.float32, [None], name = "y")

        self.fc_layer = []
        self.fc_layer.append(tf.layers.dropout(fully_connected(self.input_data, layer_connections), rate=hold_rate))
        for layer in range(fc_layers):
            self.fc_layer.append(tf.layers.dropout(fully_connected(self.fc_layer[layer-1], layer_connections), rate=hold_rate))
        
        # Output layer represents the expected reward for each possible action:  
        self.output_layer = fully_connected(self.fc_layer[fc_layers-1], self.env.action_space.n)

        # Predicted action for the best Q-value:
        self.best_q = tf.reduce_max(self.output_layer)

        # Loss function and optimize node for training:
        self.out_actions = []
        self.loss = []
        self.optimize = [] 
        # Seperate loss/optimisation for each action:
        i = 0
        for action_q in tf.split(self.output_layer, num_or_size_splits=self.env.action_space.n, axis=1):
            self.out_actions.append(action_q)
            self.loss.append(tf.squared_difference(self.target, action_q))
            self.optimize.append(tf.train.AdamOptimizer(self.lr).minimize(self.loss[i]))
            i += 1

    # Pass in the full list of sequence data and this will preprocess into the p_buffer, so that it can be fed to the graph
    def process_sequence(self, sequence_data):
        # Processed sequence needs to be the same size as the input:
        p_size = self.input_size
        p_buffer = np.zeros(shape=p_size)
        i = 0
        # Loop backwards through p_buffer:
        while(i < p_size):
            if(i < len(sequence_data)):
                i += 1
                p_buffer[p_size-i] = sequence_data[len(sequence_data)-i]
            else:
                # For the sake of simplicity, I will fill the remaining buffer with 0's:
                for index in range(p_size - i):
                    p_buffer[index] = 0
                i = p_size

        return p_buffer.reshape((-1, len(p_buffer)))

    def get_best_action(self, x_input):
        output = self.sess.run(self.output_layer, feed_dict={self.input_data: x_input})
        return np.argmax(output, axis=1)[0]

    def get_output(self, x_input):
        return self.sess.run(self.output_layer, feed_dict={self.input_data: x_input})

    def get_best_q_value(self, x_input):
        q_value = self.sess.run(self.best_q, feed_dict={self.input_data: x_input})
        return q_value

    # This will append the memory with state/action/reward data and return the average score across games:
    def get_samples(self, stop_after_limit=True, n_games=500000, max_t=250, epsilon=1.0, display_frames=False):
        game_counter = 0
        scores = []
        frames = 0
        for game in range(n_games):
            game_counter += 1
            initial_state = self.env.reset()
            sequence = [] # Sequence of states
            sequence.extend(initial_state)
            score = 0.0
            # Auto-reset after this:
            for t in range(max_t):
                frames += 1
                action = None
                processed_current_state = self.process_sequence(sequence)
                if(random.random() < epsilon):
                    action = self.env.action_space.sample()
                else:
                    # Best action for this processed sequence(acording to the model):
                    action = self.get_best_action(processed_current_state)
                # Get the reward/state information after taking this action:
                next_state, reward, done, infom = self.env.step(action)
                if(display_frames):
                    self.env.render()
                score += reward
                if(not done):
                    sequence.extend(np.concatenate((self.action_space[action], next_state)).tolist()) 
                processed_next_state = self.process_sequence(sequence)
                # Append to memory (up to limit):
                if(len(self.memory) < self.memory_size):     
                    self.memory.append((processed_current_state, action, reward, processed_next_state, done))
                elif(self.memory_index < self.memory_size):
                    # Overwrite from the beginning(when full):
                    self.memory[self.memory_index] = (processed_current_state, action, reward, processed_next_state, done)
                    self.memory_index += 1
                else:
                    self.memory[0] = (processed_current_state, action, reward, processed_next_state)
                    self.memory_index = 1

                if(done): # Game over
                    break
            scores.append(score)
            
            if(stop_after_limit):
                if(len(self.memory) >= self.memory_size):
                    break 
        return mean(scores), frames
        
    # This will attempt to train the graph:
    def train_network(self, target_mean_score = 185.1, games=200000, batch_size=32, initial_epsilon=1.0, final_epsilon=0.1, epsilon_frames_range=100000, gamma=0.95):
        while(len(self.memory) < batch_size): # Play randomly until memory has at least batch_size entries
            self.get_samples(False, 1, epsilon=1.0) 
        total_frames = 0
        scores = []
        for game in range(games):
            # Scales linearly from initial to final epsilon up to epsilon frames range:
            if(total_frames < epsilon_frames_range):
                e = initial_epsilon - ((initial_epsilon - final_epsilon)*(total_frames/epsilon_frames_range))
            else:
                e = final_epsilon
            score, frames = self.get_samples(False, 1, epsilon=e) # Play a random game, and record data to memory buffer
            scores.append(score)
            m_score = mean(scores)
            if(len(scores) == 250):
                scores = []
            total_frames += frames
            print('Game: {} Score: {}, Mean: {:.2f} Frames: {} e: {:.4f}'.format(game, score, m_score, total_frames, e))
            
            if((m_score >= target_mean_score) and (len(scores) > 100)):
                print('Target mean score reached')
                break
            mini_batch = random.sample(self.memory, batch_size)
            for p_state, action, reward, p_next_state, done in mini_batch:
                target = reward
                if(not done):
                    target = reward + gamma*self.get_best_q_value(p_next_state)
                
                # Train:
                # Seperate optimize for each loss function/action:
                self.sess.run(self.optimize[action], feed_dict={self.input_data: p_state, self.target: [target]})
        print('Training complete')
        save_path = self.saver.save(self.sess, 'saved/model.ckpt')
        print('Model saved: {}'.format(save_path))

    def load_model(self, path):
        print('Attempting to load previously saved model...')
        self.saver.restore(self.sess, path)

    # Play and record scores with the currently loaded graph:     
    def play(self, games=100, epsilo=0.05, show_game=True):
        scores = []
        total_frames = 0
        for game in range(games):
            game_num = game+1
            score, frames = self.get_samples(False, 1, epsilon=epsilo, display_frames=show_game)
            scores.append(score)
            m_score = mean(scores)
            med_score = median(scores)

            total_frames += frames
            print('Games: {} Score: {}, Mean Score: {:.2f}, Median: {:.2f} Total Frames: {}'.format(game_num, score, m_score, med_score, total_frames))

