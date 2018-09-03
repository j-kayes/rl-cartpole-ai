import tensorflow as tf
import numpy as np
import random

def fully_connected(input_tensor, outputs, activation_func=tf.nn.relu):
    result = tf.layers.dense(inputs = input_tensor,
                            units = outputs,
                            activation = activation_func,
                            kernel_initializer = tf.contrib.layers.xavier_initializer())
    return result

class Agent: 
    
    def __init__(self, environment, state_size, learning_rate, sequence_length=4, memory_size=100000):
        self.env = environment
        self.sequence_length = sequence_length
        self.state_size = state_size

        self.action_space = np.zeros(shape=(self.env.action_space.n,self.env.action_space.n))
        n = 0
        for action in self.action_space:
            action[n] = 1
            n += 1
        # Input size will be the size of the previous sequence/action buffer:
        self.input_size = sequence_length*(self.env.action_space.n + self.state_size) 

        self.memory = []
        self.memory_size = memory_size
        self.memory_index = 0
        
        self.lr = learning_rate
        
        self.build_model()
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.output_layer))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def build_model(self):
        # This will be used as the Q*(S,A) estimator, the input is the state and actions performed, and the output is the expected reward:
        self.input_data = tf.placeholder(tf.float32, [None, self.input_size], name = "x")
        self.target = tf.placeholder(tf.float32, [None], name = "y")

        self.fc1 = tf.layers.dropout(fully_connected(self.input_data, 128), rate=0.8)
        self.fc2 = tf.layers.dropout(fully_connected(self.fc1, 128), rate=0.8)
        self.fc3 = tf.layers.dropout(fully_connected(self.fc2, 128), rate=0.8)
        self.fc4 = tf.layers.dropout(fully_connected(self.fc3, 128), rate=0.8)
        self.fc5 = tf.layers.dropout(fully_connected(self.fc4, 128), rate=0.8)

        # Output layer represents the expected reward:  
        self.output_layer = fully_connected(self.fc5, 1)

    # Pass in the full list of sequence data and this will preprocess into the p_buffer, so that it can be fed to the graph
    def process_sequence(self, sequence_data):
        # Processed sequence will be of the same size as the input, without the final input data:
        p_size = self.input_size-self.env.action_space.n
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

        return p_buffer

    def get_output(self, x_input):
        return self.sess.run(self.output_layer, feed_dict={self.input_data: x_input})

    # Loop over possible actions to find the action with the best expected reward acording to the model:
    def get_best_action(self, sequence_data):
        best_action = None
        best_reward = None
        for action in self.action_space:
            # Join with processed state/action sequence data:
            x_input = np.concatenate((self.process_sequence(sequence_data), action))
            # Feed into graph, determine reward:
            exp_reward = self.get_output(x_input.reshape((-1, len(x_input))))
            if((best_reward is None) or (exp_reward > best_reward)):
                best_reward = exp_reward
                best_action = action
        return best_action

    # This will append the memory with state/action/reward data:
    def get_samples(self, stop_after_limit=True, n_games=500000, max_t=250, epsilon=1.0):
        game_counter = 0
        for game in range(n_games):
            game_counter += 1
            initial_state = self.env.reset()
            sequence = [] # Sequence of states
            sequence.extend(initial_state)
            # Auto-reset after this:
            for t in range(max_t):
                action = None
                score = 0
                processed_current_state = self.process_sequence(sequence)
                if(random.random() < epsilon):
                    action = self.env.action_space.sample()
                else:
                    # Best action for this processed sequence(acording to the model):
                    action = self.get_best_action(processed_current_state)
                # Get the reward/state information after taking this action:
                next_state, reward, done, infom = self.env.step(action)
                score += reward
                con_array = np.concatenate((self.action_space[action], next_state))
                sequence.extend(con_array.tolist()) # For extending the sequence to process
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
            
            if(stop_after_limit):
                print('After game {} memory buffer {}% full'.format(game_counter, 100.0*len(self.memory)/self.memory_size))
                if(len(self.memory) >= self.memory_size):
                    break 
            else:
                print('Memory buffer {}% full, score for game: {}'.format(100.0*len(self.memory)/self.memory_size, score))

    # This will attempt to train the graph:
    def train_network(self, games = 5000, batch_size=32, epochs=5, initial_epsilon=1.0, gamma=0.95):
        while(len(self.memory) < batch_size):
            self.get_samples(False, 1, epsilon=1.0) # Play randomly until memmory has at least batch_size entries
        # TODO: Check accuracy/score during and after training:
        for epoch in range(epochs):
            for game in range(games):
                print('Game number ', game)
                self.get_samples(False, 1, epsilon=initial_epsilon) # Play a random game, and record data to memory buffer
                mini_batch = random.sample(self.memory, batch_size)
                for state, action, reward, next_state, done in mini_batch:
                    target = reward
                    if(not done):
                        next_best_action = self.get_best_action(next_state)
                        input_full = np.concatenate((next_state, next_best_action))
                        target = reward + float(gamma*self.get_output(input_full.reshape((-1, len(input_full))))) 
                    x_input = np.concatenate((state, self.action_space[action])) # Need to concatenate the action taken for input to be the corrent length
                    #y_output = self.sess.run(self.output_layer, feed_dict={self.input_data: x_input.reshape((-1, len(x_input)))})
                    
                    # Train:
                    self.sess.run(self.optimize, feed_dict={self.input_data: x_input.reshape((-1, len(x_input))), self.target: np.array([target])})
