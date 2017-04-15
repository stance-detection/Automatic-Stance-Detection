import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import sys
from tensorflow.contrib.rnn import OutputProjectionWrapper

class BatchSampler(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def batches(self, batch_size):
        ids = list(range(self.y.shape[0]))
        np.random.shuffle(ids)
        
        results = []
        for i in range(0, int(np.ceil(len(ids)/batch_size))):
            batch_ids = ids[i*batch_size:np.min([len(ids), (i+1)*batch_size])]
            X_batch = self.X[batch_ids]
            X_headlines = X_batch[:,0]
            X_bodies = X_batch[:,1]
            y_batch = self.y[batch_ids]
            results.append((X_headlines, X_bodies, y_batch))
        return results

class LSTM(object):
    def __init__(self, X_train, y_train):
        feature_size = 300
        hidden_size = 100
        output_size = 4
        learning_rate = 0.01
        num_epochs = 2
        batch_size = 50
        #batch_size = 1
        TINY = 1e-6 # to avoid NaNs in logs
      
        #Setup  
        self.headline_input = tf.placeholder(tf.float32, [batch_size, None, feature_size]) #[batch_size, time, in]
        self.body_input = tf.placeholder(tf.float32, [batch_size, None, feature_size])
        self.gold = tf.placeholder(tf.float32, [batch_size, output_size])
        
        with tf.variable_scope('headline'):
            self.headline_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        with tf.variable_scope('body'):
            self.body_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        
        self.initial_state = self.headline_lstm.zero_state(batch_size, tf.float32)

        self.weights = tf.Variable(tf.random_normal([hidden_size, output_size], dtype=tf.float32))
        self.biases = tf.Variable(tf.random_normal([output_size], dtype=tf.float32))


        with tf.variable_scope('headline'):
            self.headline_outputs, self.headline_states = tf.nn.dynamic_rnn(self.headline_lstm, self.headline_input, initial_state=self.initial_state)

        with tf.variable_scope('body'):
            #initial_body_state = (headline_states[0], body_lstm.zero_state(batch_size, tf.float32)[1])  # (c, h)
            #initial_body_state = body_lstm.zero_state(batch_size, tf.float32)
            self.initial_body_state = self.headline_states
            self.body_outputs, self.body_states = tf.nn.dynamic_rnn(self.body_lstm, self.body_input, initial_state=self.initial_body_state)

        outputs = tf.matmul(self.body_outputs[:, -1, :], self.weights) + self.biases
        self.predicted_stance = tf.nn.softmax(outputs)
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predicted_stance, labels = self.gold))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.error)



        #Train
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        
        batcher = BatchSampler(X_train, y_train)
        
        for epoch in range(num_epochs):
            epoch_error = 0
            for (X_headlines, X_bodies, y) in batcher.batches(batch_size):
                results = self.session.run(self.predicted_stance, feed_dict={
                                                               self.headline_input: X_headlines,
                                                               self.body_input: X_bodies,
                                                               self.gold: y,
                                                               })
                print("Epoch %d" % (epoch))

def test(self, X_test, y_test):
    y_pred = self.session.run(self.predicted_stance, feed_dict={
                                           self.headline_input: X_test[:,0],
                                           self.body_input:X_test[:,1], 
                                           self.gold: y_test,
            })
    return y_pred

