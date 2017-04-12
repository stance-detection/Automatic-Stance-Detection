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
        for i in range(0, np.ceil(len(ids)/batch_size)):
            batch_ids = ids[i*batch_size:np.min(len(ids), (i+1)*batch_size)]
            X_batch = [self.X[0][batch_ids], self.X[1][batch_ids]]
            y_batch = self.y[batch_ids]
            results.append((X_batch, y_batch))
        return results

class LSTM(object):
    def __init__(self, X_train, y_train):
        feature_size = 300
        hidden_size = 100
        output_size = 3
        learning_rate = '0.01'
        num_epochs = ''
        epoch_size = ''
        batch_size = 50
        TINY = 1e-6 # to avoid NaNs in logs
        
        self.headline_input = tf.placeholder(tf.float32, [batch_size, None, feature_size]) #[batch_size, time, in}
        self.body_input = tf.placeholder(tf.float32, [batch_size, None, feature_size])
        self.gold = tf.placeholder(tf.float32, [batch_size, output_size])
        
        headline_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        body_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        
        # Initial state of the LSTM memory.
        initial_state = headline_lstm.zero_state(batch_size, tf.float32)
        
        #time-major means time is first dimension of input
        # inputs = tf.unstack(self.headline_input, axis=?)
        headline_outputs, headline_states = tf.nn.dynamic_rnn(headline_lstm, self.headline_input, initial_state=initial_state)
        projection_wrapper = OutputProjectionWrapper(body_lstm, output_size)
        #body_outputs, body_states = tf.nn.dynamic_rnn(projection_wrapper, self.body_input, initial_state=headline_states[-1])
        old_state = headline_states[-1]
        print(old_state.shape)

        tf.nn.dynamic_rnn(projection_wrapper, self.body_input, initial_state=old_state)
        sys.exit()
        
        self.predicted_stance = tf.nn.softmax(body_outputs[-1])
        error = -(self.gold * tf.log(self.predicted_stance + TINY) + (1.0 - self.gold) * tf.log(1.0 - self.predicted_stance + TINY))
        error = tf.reduce_mean(error)
        
        train_fn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)
        self.accuracy = tf.reduce_mean(tf.cast(tf.abs(self.gold - self.predicted_stance) < 0.5, tf.float32))
        
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        
        batcher = BatchSampler(X_train, y_train)
        
        for epoch in range(num_epochs):
            epoch_error = 0
            for X, y in batcher.batches(batch_size):
                epoch_error += self.session.run([error, train_fn], {
                                                               self.headline_input: X[0],
                                                               self.body_input: X[1],
                                                               self.gold: y,
                                                               })[0]
                epoch_error /= epoch_size
                print("Epoch %d, train error: %.2f" % (epoch, epoch_error))

def test(self, X_test, y_test):
    y_pred = self.session.run(self.predicted_stance, {
                                           self.headline_input: X_test[0],
                                           self.body_input:X_test[1], 
                                           self.gold: y_test,
            })
    return y_pred

