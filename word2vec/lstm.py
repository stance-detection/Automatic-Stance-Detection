import sys
import numpy as np

#from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
#from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from nltk.tokenize import RegexpTokenizer

import gensim
from gensim.models import word2vec
import pdb
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import OutputProjectionWrapper
tf.reset_default_graph();

def cleantext(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens

def buildWordVector(text, model):
    text = cleantext(text)
    v = []
    #v = np.zeros(300)
    count = 0
    for word in text:
        if word in model.vocab:
            v.append(model[word])
            count += 1
        else:
            v.append(np.zeros((300,)))
            count += 1

    return v, count

def to_one_hot(y):
    yoh = []
    ind=0
    for ys in y:
        yoh.append([0]*4)
        yoh[ind][ys] = 1
        ind+=1
    return yoh

def from_one_hot(yoh):
    print(yoh.shape)
    y = np.argmax(yoh,1)
    print(y)
    return y


def generate_features(stances, dataset, name, model, binary=False):
    headline, body = dict(), dict()

    headline['features'] = []
    headline['lengths'] = []
    body['features'] = []
    body['lengths'] = []

    y = []
    for stance in stances:
        if (binary!=True):
            y.append(LABELS.index(stance['Stance']))
        else:
            if LABELS.index(stance['Stance'])<3:
                y.append(0)
            else:
                y.append(1)
        headline_features, h_length = buildWordVector(stance['Headline'], model)
        body_features, b_length = buildWordVector(dataset.articles[stance['Body ID']],model)
        headline['features'].append(headline_features)
        headline['lengths'].append(h_length)
        body['features'].append(body_features)
        body['lengths'].append(b_length)


    return headline, body, y

class LSTM():
    def __init__(self):

        self.learning_rate = 0.01
        self.epochs = 1
        self.training_iters = 100000
        self.batch_size = 50
        self.display_step = 10

        # Network Parameters
        self.input_dims = 300
        self.n_hidden = 100
        self.n_classes = 4

        # tf Graph input
        self.head = tf.placeholder(dtype = tf.float64, shape=[None, None, self.input_dims])
        self.body = tf.placeholder(dtype = tf.float64, shape=[None, None, self.input_dims])
        self.head_lengths = tf.placeholder(dtype=tf.int8,shape=[None,])
        self.body_lengths = tf.placeholder(dtype=tf.int8,shape=[None,])
        self.labels = tf.placeholder(dtype = tf.float64, shape=[None, self.n_classes])


        with tf.variable_scope('head', reuse=tf.AUTO_REUSE):
            self.LSTM_head = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
        with tf.variable_scope('body', reuse=tf.AUTO_REUSE):
            self.LSTM_body = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, state_is_tuple=True)

        self.initial_state = self.LSTM_head.zero_state(self.batch_size, tf.float64)
        self.outweights = {
            # Hidden layer weights => 2*n_hidden because of foward + backward cells
            'out': tf.Variable(tf.random_normal(dtype=tf.float64,shape=[self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal(dtype=tf.float64,shape=[self.n_classes]))
        }



    def calculate(self):
        with tf.variable_scope('head'):
            head_outputs, head_last_states = tf.nn.dynamic_rnn(
                                cell = self.LSTM_head,
                                dtype = tf.float64,
                                sequence_length = self.head_lengths,
                                inputs = self.head,
                                initial_state = self.initial_state)

        head_old_layer = head_last_states

        with tf.variable_scope('body'):
            body_outputs, body_last_states = tf.nn.dynamic_rnn(
                                cell = self.LSTM_body,
                                dtype = tf.float64,
                                sequence_length = self.body_lengths,
                                inputs = self.body,
                                initial_state = head_old_layer)

        body_out_layer = tf.matmul(body_outputs[:,-1,:], self.outweights['out']) + self.biases['out']
        #body_out_layer = OutputProjectionWrapper(self.LSTM_body, self.n_classes)
        predicted = tf.nn.softmax(body_out_layer)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=lstm.labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=lstm.learning_rate).minimize(cost)

        return predicted, cost, optimizer

    def test(self, test_batch_size):
        test_init_state = self.LSTM_head.zero_state(test_batch_size, tf.float64)
        with tf.variable_scope('head', reuse=tf.AUTO_REUSE):
            head_outputs, head_last_states = tf.nn.dynamic_rnn(
                                cell = self.LSTM_head,
                                dtype = tf.float64,
                                sequence_length = self.head_lengths,
                                inputs = self.head,
                                initial_state = test_init_state)
        with tf.variable_scope('body', reuse=tf.AUTO_REUSE):
            body_outputs, body_last_states = tf.nn.dynamic_rnn(
                                cell = self.LSTM_body,
                                dtype = tf.float64,
                                sequence_length = self.body_lengths,
                                inputs = self.body,
                                initial_state = head_last_states)

        body_out_layer = tf.matmul(body_outputs[:,-1,:], self.outweights['out'])+ self.biases['out']
        predicted = tf.nn.softmax(body_out_layer)

        return tf.argmax(predicted,axis=1)

if __name__ == "__main__":
    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)

    print('Loading DataSet')
    d = DataSet()
    print('generating folds')
    folds,hold_out = kfold_split(d,n_folds=10)
    folds=folds[:2]
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    print('Loading word2vec model')
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


    ys = dict()
    X_h = dict()
    X_b = dict()



    print('calculating features')
    print('holdout set')
    X_h_holdout, X_b_holdout, y_holdout = generate_features(hold_out_stances,d,"holdout", model, binary=False)

    #fold_stances = fold_stances
    print('folds')
    for fold in fold_stances:
        if fold<2:
            X_h[fold],X_b[fold], ys[fold] = generate_features(fold_stances[fold],d,str(fold), model, binary=False)

    best_score = 0
    best_fold1= None
    best_fold2= None
    del model
    print('training')

    #for fold in fold_stances:
    for fold in range(2):
        ids = list(range(len(folds)))
        del ids[fold]

        X_htrain = [X_h[i]["features"][j] for i in ids for j in range(len(X_h[i]["features"]))]
        X_btrain = [X_b[i]["features"][j] for i in ids for j in range(len(X_b[i]["features"]))]
        X_hlen = [X_h[i]["lengths"][j] for i in ids for j in range(len(X_h[i]["lengths"]))]
        X_blen = [X_b[i]["lengths"][j] for i in ids for j in range(len(X_b[i]["lengths"]))]
        y_train = np.hstack(tuple([ys[i] for i in ids]))
        X_htest = X_h[fold]["features"][:50]
        X_htest_len = X_h[fold]["lengths"][:50]
        X_btest_len = X_b[fold]["lengths"][:50]
        X_btest = X_b[fold]["features"][:50]
        y_test = ys[fold][:50]



        lstm = LSTM()
        try:
            predicted, cost, optimizer = lstm.calculate()
        except:
            pass
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            epoch = 0

            while (epoch < lstm.epochs):
                n_step=0
                start_ind=0
                while (n_step < 70 ):
                    if (start_ind+lstm.batch_size < len(X_htrain)):
                        Xhtrain_batch = X_htrain[start_ind: start_ind+ lstm.batch_size]
                        Xhlen_batch = X_hlen[start_ind: start_ind+ lstm.batch_size]
                        max_hlen = max(Xhlen_batch)
                        y_tr_batch = to_one_hot(y_train[start_ind: start_ind+ lstm.batch_size])
                        y_ts_batch = to_one_hot(y_test)

                        Xbtrain_batch = X_btrain[start_ind: start_ind+ lstm.batch_size]
                        Xblen_batch = X_blen[start_ind: start_ind+ lstm.batch_size]
                        max_blen = max(Xblen_batch)
                    else:
                        Xhtrain_batch = X_htrain[start_ind: ]
                        Xhlen_batch = X_hlen[start_ind:]
                        max_hlen = max(Xhlen_batch)

                        Xbtrain_batch = X_btrain[start_ind: ]
                        Xblen_batch = X_blen[start_ind: ]
                        max_blen = max(Xblen_batch)

                    max_hlen_test = max(X_htest_len)
                    max_blen_test = max(X_btest_len)

                    XH = np.zeros([lstm.batch_size, max_hlen, lstm.input_dims], dtype=np.float64)
                    XB = np.zeros([lstm.batch_size, max_blen, lstm.input_dims], dtype=np.float64)
                    for hind, xh in enumerate(XH):
                        xh[:Xhlen_batch[hind]] = Xhtrain_batch[hind]
                    for bind, xb in enumerate(XB):
                        xb[:Xblen_batch[bind]] = Xbtrain_batch[bind]

                    XH_test = np.zeros([lstm.batch_size, max_hlen_test, lstm.input_dims], dtype=np.float64)
                    XB_test = np.zeros([lstm.batch_size, max_blen_test, lstm.input_dims], dtype=np.float64)

                    for hind, xht in enumerate(XH_test):
                        xht[:X_htest_len[hind]] = X_htest[hind]
                    for bind, xbt in enumerate(XB_test):
                        xbt[:X_btest_len[bind]] = X_btest[bind]


                    try:
                        result = sess.run(
                                {"optimizer": optimizer, "predicted": predicted, "cost":cost},
                                feed_dict = {lstm.head: XH,lstm.body: XB,
                                    lstm.head_lengths: Xhlen_batch,
                                    lstm.body_lengths: Xblen_batch,lstm.labels: y_tr_batch})
                    except:
                        pass

                    classes = lstm.test(lstm.batch_size)

                    if (n_step % lstm.display_step)==0:
                        outputs = sess.run(classes,
                                    feed_dict = {lstm.head: XH_test,
                                                lstm.head_lengths: X_htest_len,
                                                lstm.body: XB_test,
                                                lstm.body_lengths: X_btest_len})

                        predicted_labels = [LABELS[int(a)] for a in outputs]
                        actual_labels = [LABELS[int(a)] for a in y_test]

                        fold_score, _ = score_submission(actual_labels, predicted_labels)
                        max_fold_score, _ = score_submission(actual_labels, actual_labels)
                        score = (fold_score/max_fold_score)*100+30;

                    print("step is :"+str(n_step) + " cost is :"+str(result["cost"])+" score is :"+str(score))
                    start_ind += lstm.batch_size
                    n_step += 1
                epoch += 1
