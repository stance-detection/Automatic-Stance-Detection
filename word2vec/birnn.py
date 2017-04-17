import sys
import numpy as np

from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import gensim
from gensim.models import word2vec

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import OutputProjectionWrapper


def cleantext(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens


def buildWordVector(text, model, mode):

    #to concatenate word2vec and glove embeddeings, use np.hstack((a,b))
    #but looking at the results of glove, this might not be helpful

    text = cleantext(text)
    if mode=='word2vec':
        v = np.zeros(300)
        count = 0
        for word in text:
            if word in model.vocab:
                v = v + model[word]
                count += 1
        if count!= 0:
            v /= count

    elif mode=='glove100':
        v = np.zeros(100)
        count = 0
        for word in text:
            if word in model:
                v = v + np.asarray(model[word.lower()])
                count += 1
        if count!=0:
            v /= count

    elif mode=='glove200':
        v = np.zeros(200)
        count = 0
        for word in text:
            if word in model:
                v = v + np.asarray(model[word.lower()])
                count += 1
        if count!=0:
            v /= count

    elif mode=='glove300':
        v = np.zeros(300)
        count = 0
        for word in text:
            if word in model:
                v = v + np.asarray(model[word.lower()])
                count += 1
        if count!=0:
            v /= count

    elif mode=='glove50':
        v = np.zeros(50)
        count = 0
        for word in text:
            if word in model:
                v = v + np.asarray(model[word.lower()])
                count += 1
        if count!=0:
            v /= count

    else:
        return None

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
#    print(yoh.shape)
    y = np.argmax(yoh,1)
#    print(type(y))
#    print(y)
    return list(y)


def generate_baseline_features(stances, dataset, name, binary=True):
    h, b, y = [],[],[]
    baseline_dir = '../baseline/'
    for stance in stances:
        if (binary!=True):
            y.append(LABELS.index(stance['Stance']))
        else:
            if LABELS.index(stance['Stance'])<3:
                y.append(0)
            else:
                y.append(1)

        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, baseline_dir+ "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, baseline_dir+ "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, baseline_dir+ "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, baseline_dir+ "features/hand."+name+".npy")
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X, y


def generate_features(stances, dataset, name, model, mode, binary=False):
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
        headline_features, h_length = buildWordVector(stance['Headline'], model, mode)
        body_features, b_length = buildWordVector(dataset.articles[stance['Body ID']],model, mode)
        headline['features'].append(headline_features)
        headline['lengths'].append(h_length)        
        body['features'].append(body_features)
        body['lengths'].append(b_length)


    return headline, body, y

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    #Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden)#, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



       
    


if __name__ == "__main__":

    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)

    print('Loading DataSet')
    d = DataSet()
    print('generating folds')
    folds,hold_out = kfold_split(d,n_folds=10)
    #folds=folds[:2]
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    print('Loading word2vec model')
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    mode_dict = ['word2vec','glove50','glove100','glove200','glove300']

    for dic_ind in range(1):
        mode = mode_dict[dic_ind]
        if mode == 'word2vec':
            model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        elif mode== 'glove50':
            gloveFile = 'glove.6B.50d.txt'
            model = loadGloveModel(gloveFile)
        elif mode== 'glove100':
            gloveFile = 'glove.6B.100d.txt'
            model = loadGloveModel(gloveFile)
        elif mode== 'glove200':
            gloveFile = 'glove.6B.200d.txt'
            model = loadGloveModel(gloveFile)
        elif mode== 'glove300':
            gloveFile = 'glove.6B.300d.txt'
            model = loadGloveModel(gloveFile)




    ys = dict()
    X_h = dict()
    X_b = dict()



    print('calculating features')
    print('holdout set')
    X_h_holdout, X_b_holdout, y_holdout = generate_features(hold_out_stances,d,"holdout", model, mode, binary=False)

    #fold_stances = fold_stances
    print('folds')
    for fold in fold_stances:
        if fold<10:
            X_h[fold],X_b[fold], ys[fold] = generate_features(fold_stances[fold],d,str(fold), model, mode, binary=False)

    best_score = 0
    best_fold1= None
    best_fold2= None
    del model
    print('training')

    learning_rate = 0.001
    training_iters = 100000
    batch_size = 128
    display_step = 10
    epochs = 50

    # Network Parameters
    n_input = 300 # MNIST data input (img shape: 28*28)
    n_steps = 2 # timesteps
    n_hidden = 128 # hidden layer num of features
    n_classes = 4 # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    labels = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = RNN(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    #for fold in fold_stances:
    for fold in range(10):
        ids = list(range(len(folds)))
        del ids[fold]

        X_htrain = [X_h[i]["features"][j] for i in ids for j in range(len(X_h[i]["features"]))]
        X_btrain = [X_b[i]["features"][j] for i in ids for j in range(len(X_b[i]["features"]))]
        y_train = np.hstack(tuple([ys[i] for i in ids]))
        X_htest = X_h[fold]["features"]
        X_btest = X_b[fold]["features"]
        y_test = ys[fold]
        
        
        with tf.Session() as sess:
            sess.run(init)
            epoch = 0
            
            while (epoch < epochs):
                n_step=0
                print( "epoch: ", epoch)
                start_ind=0
                while (n_step*batch_size < len(X_htrain)):
                
                    if (start_ind+batch_size < len(X_htrain)):
                        Xhtrain_batch = X_htrain[start_ind: start_ind+ batch_size]
                        y_tr_batch = to_one_hot(y_train[start_ind: start_ind+ batch_size])
                        Xbtrain_batch = X_btrain[start_ind: start_ind+ batch_size]
                    

                    train = np.zeros((batch_size, n_steps, n_input))
                    for batch_ind in range(batch_size):
                        train[batch_ind, 0, :] = Xhtrain_batch[batch_ind]
                        train[batch_ind, 1, :] = Xbtrain_batch[batch_ind]



                    result = sess.run(
                            {"optimizer": optimizer, "predicted": pred, "cost":cost},
                            feed_dict = {x: train,labels: y_tr_batch})
                    if n_step % display_step == 0:
                        predic = sess.run(pred, feed_dict={x: train, labels: y_tr_batch})
                
                        predic_lab = [LABELS[int(a)] for a in from_one_hot(predic)]
                        actual = [LABELS[int(a)] for a in np.argmax(y_tr_batch,1)]

                    start_ind += batch_size
                    n_step += 1
                epoch += 1
            test = np.zeros((len(X_htest),n_steps, n_input))
            for test_ind in range(len(X_htest)):
                test[test_ind, 0, :] = X_htest[test_ind]
                test[test_ind, 0, :] = X_btest[test_ind]


            predic = sess.run(pred, feed_dict={x: test, labels: to_one_hot(y_test)})
            predic_lab = [LABELS[int(a)] for a in from_one_hot(predic)]
            actual = [LABELS[int(a)] for a in np.argmax(to_one_hot(y_test),1)]
            print('confusion matrix')
            report_score(actual, predic_lab)

