import sys
import numpy as np

#from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
#from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from nltk.tokenize import RegexpTokenizer
#from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
#from sklearn.linear_model import LogisticRegression

import gensim
from gensim.models import word2vec

import tensorflow as tf
from tensorflow.contrib import rnn



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

        self.learning_rate = 0.001
        self.epochs = 50
        self.training_iters = 100000
        self.batch_size = 50
        self.display_step = 10

        # Network Parameters
        self.input_dims = 300
        self.n_hidden = 100
        self.n_classes = 4 

        # tf Graph input
        self.head = tf.placeholder(dtype = tf.float64, shape=[self.batch_size, None, self.input_dims])
        self.body = tf.placeholder(dtype = tf.float64, shape=[self.batch_size, None, self.input_dims])
        self.head_lengths = tf.placeholder(dtype=tf.int8,shape=[self.batch_size,])
        self.body_lengths = tf.placeholder(dtype=tf.int8,shape=[self.batch_size,])
        self.labels = tf.placeholder(dtype = tf.float64, shape=[self.batch_size, self.n_classes])

        self.LSTM_head = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.LSTM_body = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

        self.initial_state = self.LSTM_head.zero_state(self.batch_size, tf.float64)

       
    def calculate(self):
        head_outputs, head_last_states = tf.nn.dynamic_rnn(
                                cell = self.LSTM_head,
                                dtype = tf.float64,
                                sequence_length = self.head_lengths,
                                inputs = self.head,
                                initial_state = self.initial_state)

        head_old_layer = head_last_states[-1]
        
        body_outputs, body_last_states = tf.nn.dynamic_rnn(
                                cell = self.LSTM_body,
                                dtype = tf.float64,
                                sequence_length = self.body_lengths,
                                inputs = self.body,
                                initial_state = head_old_layer)
        

        body_out_layer = OutputProjectionLayer(self.LSTM_body, self.n_classes)
        predicted = tf.nn.softmax(body_out_layer)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=lstm.labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=lstm.learning_rate).minimize(cost)
        


        return predicted, cost, optimizer
        


    


if __name__ == "__main__":

    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)

    print('Loading DataSet')
    d = DataSet()
    print('generating folds')
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    print('Loading word2vec model')
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


    ys = dict()
    X_h = dict()
    X_b = dict()



    print('calculating features')
    print('holdout set')
    X_h_holdout, X_b_holdout, y_holdout = generate_features(hold_out_stances,d,"holdout", model, binary=False)


    print('folds')
    for fold in fold_stances:
        X_h[fold],X_b[fold], ys[fold] = generate_features(fold_stances[fold],d,str(fold), model, binary=False)



    best_score = 0
    best_fold1= None
    best_fold2= None
    del model
    print('training')

    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_htrain = [X_h[i]["features"][j] for i in ids for j in range(len(X_h[i]["features"]))]
        X_btrain = [X_b[i]["features"][j] for i in ids for j in range(len(X_b[i]["features"]))]
        X_hlen = [X_h[i]["lengths"][j] for i in ids for j in range(len(X_h[i]["lengths"]))]
        X_blen = [X_b[i]["lengths"][j] for i in ids for j in range(len(X_b[i]["lengths"]))]

        y_train = np.hstack(tuple([ys[i] for i in ids]))
        
        X_htest = X_h[fold]["features"]
        X_btest = X_b[fold]["features"]
        y_test = ys[fold]
        


        lstm = LSTM()
        predicted, cost, optimizer = lstm.calculate()
        


        with tf.Session() as sess:
            sess.run(init)
            epoch = 0
            start_ind=0

            while (epoch < lstm.epochs):
                #indices = get_next_batch(len(X_htrain))                            
            #reshape a list of lists as tensors and use dynamic padding
            #fetch each batch
                n_step=0
                while n_step



                batched_Xhead = tf.train.batch(
                                tensors = X_htrain[start_ind: start_ind+ lstm.batch_size],
                                batch_size = lstm.batch_size,
                                dynamic_pad = True,
                                allow_smaller_final_batch=True)

                batched_Xbody = tf.train.batch(
                                tensors = ,
                                batch_size = lstm.batch_size,
                                dynamic_pad = True,
                                allow_smaller_final_batch=True)

    
                batched_Xhead_lengths = tf.train.batch(
                                tensors = ,
                                batch_size = lstm.batch_size,
                                dynamic_pad = True,
                                allow_smaller_final_batch=True)

        
                batched_Xbody_lengths = tf.train.batch(
                                tensors = ,
                                batch_size = lstm.batch_size,
                                dynamic_pad = True,
                                allow_smaller_final_batch=True)

    
                batched_ylabels = tf.train.batch(
                                tensors = ,
                                batch_size = lstm.batch_size,
                                dynamic_pad = True,
                                allow_smaller_final_batch=True)
            


                resut = tf.contrib.learn.run_n(
                            {"optimizer": optimizer, "predicted": predicted, "cost":cost},
                            feed_dict = {lstm.head = Xhead,lstm.body = Xbody,
                                lstm.head_lengths = Xhead_lengths,
                                lstm.body_lengths = Xbody_lengths,lstm.labels = ylabels})
                                        








        #clf = LogisticRegression()
        clf = RandomForestClassifier(n_estimators=200, n_jobs=4, verbose = True)
        #clf = svm.SVC(kernel = 'rbf', gamma=0.5, C=1, verbose=True)
        clf.fit(X_train, y_train)


        
        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test_true]


        fold_score, _ = score_submission(actual, final)
        max_fold_score, _ = score_submission(actual, actual)
        score = fold_score/max_fold_score





        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold1 = clf
            best_fold2 = clf2


    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold1.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]
    print('confusion matrix for randomforestclassifier')
    report_score(actual,predicted)





