
import tensorflow as tf
class Agent:
    # This will be used as the Q*(S,A) estimator, the input is the state and actions performed, and the output is the expected reward:
    input_data = tf.placeholder(tf.float32, [None, 6, 1], name = "x")

    fc1 = tf.layers.dropout(fully_connected(input_data, 128), rate=0.8)
    fc2 = tf.layers.dropout(fully_connected(fc1, 128), rate=0.8)
    fc3 = tf.layers.dropout(fully_connected(fc2, 128), rate=0.8)
    fc4 = tf.layers.dropout(fully_connected(fc3, 128), rate=0.8)
    fc5 = tf.layers.dropout(fully_connected(fc4, 128), rate=0.8)

    # Output layer represents the expected reward:  
    output_layer = fully_connected(fc5, 1)
    def __init__(self):
        self.memory = []
        self.state_sequence = []
        self.sess = tf.Session()

    def deep_q_train(self):
