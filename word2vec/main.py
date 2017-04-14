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
from scipy.spatial.distance import cosine
from sklearn import svm


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

    return v



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

def generate_features(stances, dataset, name, model, mode, binary=True):
    headline, body, y = [], [], [] 
    for stance in stances:
        if (binary!=True):
            y.append(LABELS.index(stance['Stance']))
        else:
            if LABELS.index(stance['Stance'])<3:
                y.append(0)
            else:
                y.append(1)
        headline.append(buildWordVector(stance['Headline'],model, mode))
        body.append(buildWordVector(dataset.articles[stance['Body ID']], model, mode))
    concatenated = np.c_[headline,body]

    return concatenated, headline, body, y

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



#def rel_unrel_classification():






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


        Xhs = dict()
        Xbs = dict()
        ys = dict()
        Xcs = dict()
        Xhs_nb = dict()
        Xbs_nb = dict()
        ys_nb = dict()
        Xcs_nb = dict()
        fold_stances_nb = dict()
        ys_true=dict()

        X_baseline = dict()
        y_baseline = dict()

        #Xc_holdout : concatenated features of holdout set
        #Xh_holdout : headline features of holdout set
        #Xb_holdout : body features of holdout set
        #y_holdout  : true labels of holdout set
        #ys[fold]   : binary (related (0) vs unrelated (1)) labels for the stances in that fold
        #ys_true[]  : true labels of the stances in that fold
        #Xcs[fold]  : concatenated features of the stances in that fold
        #Xhs[fold]  : headline features of the stances in that fold
        #Xbs[fold]  : body features of the stances in that fold


        print('calculating features')
        print('holdout set')
        Xc_holdout, Xh_holdout, Xb_holdout, y_holdout = generate_features(hold_out_stances,d,"holdout", model, mode, binary=False)
        X_baseline_holdout, _ = generate_baseline_features(hold_out_stances, d, "holdout", binary= False)


        print('folds')
        for fold in fold_stances:
            Xcs[fold], Xhs[fold], Xbs[fold], ys[fold] = generate_features(fold_stances[fold],d,str(fold), model, mode, binary=True)
            _,_,_,ys_true[fold] = generate_features(fold_stances[fold],d,str(fold),model, mode, binary=False)
            X_baseline[fold], _ = generate_baseline_features(fold_stances[fold], d, str(fold), binary = True)



        binary_flag=dict()
        binary_ind = dict()
        for fold in fold_stances:
            binary_flag[fold] = [1 if x['Stance']=='unrelated' else 0 for x in fold_stances[fold]]
            binary_ind[fold] = [i for i,e in enumerate(binary_flag[fold]) if e==0]

        #Xc_holdout_nb, Xh_holdout_nb, Xb_holdout_nb, y_holdout_nb = generate_features(hold_out_stances,d,"holdout",model,binary=False)
        #_nb represents not binary (not just related vs unrelated), it considers the stances
        #_nb[] lists are calculated only for those data which are related in the training set
        for fold in fold_stances:
            fold_stances_nb[fold] = [fold_stances[fold][x] for x in binary_ind[fold]]
            Xcs_nb[fold], Xhs_nb[fold], Xbs_nb[fold], ys_nb[fold] = generate_features(fold_stances_nb[fold],d,str(fold), model, mode, binary=False)


        best_score = 0
        best_fold1= None
        best_fold2= None
        del model
        print('training')

        fscore = {}
        for pmtune1 in range(1,2):
            fscore[pmtune1]=[]
            for pmtune2 in range(1, pmtune1+1):
                print(pmtune1, pmtune2)
                for fold in fold_stances:
                    ids = list(range(len(folds)))
                    del ids[fold]

                    X_train = np.vstack(tuple([X_baseline[i] for i in ids]))
                    y_train = np.hstack(tuple([ys[i] for i in ids]))
                    
                    X_test = X_baseline[fold]
                    y_test = ys[fold]
                    y_test_true = ys_true[fold]
                    
                    #clf = LogisticRegression()
                    clf = RandomForestClassifier(n_estimators=200*pmtune1, n_jobs=4, verbose = False)
                    #clf = svm.SVC(kernel = 'rbf', gamma=0.5, C=1, verbose=True)
                    clf.fit(X_train, y_train)


                    X_train_nb = np.vstack(tuple([Xcs_nb[i] for i in ids]))
                    y_train_nb = np.hstack(tuple([ys_nb[i] for i in ids]))
             
                    clf2 = RandomForestClassifier(n_estimators=200*pmtune2,n_jobs=4,verbose=False)
                    clf2.fit(X_train_nb,y_train_nb)                            
                            
                    
                    predicted = [LABELS[3] if a==1 else LABELS[0] for a in clf.predict(X_test)]
                    actual = [LABELS[int(a)] for a in y_test_true]

                    init_pred = dict()
                    init_pred_ind=dict()
                    #fold_stances_new=dict()
                        
                    init_pred[fold] = [int(a) for a in clf.predict(X_baseline[fold])]
                    init_pred_ind[fold] = [i for i,e in enumerate(init_pred[fold]) if e==0]
                    
                    Xcs_temp = [Xcs[fold][x] for x in init_pred_ind[fold]]
                    predicted_new = [LABELS[int(a)] for a in clf2.predict(Xcs_temp)]
                    final = predicted
                    for i,e in enumerate(init_pred_ind[fold]):
                        final[e] = predicted_new[i]


                    fold_score, _ = score_submission(actual, final)
                    max_fold_score, _ = score_submission(actual, actual)
                    score = fold_score/max_fold_score

                    #for f in ids:

                    #    init_pred[f] = [int(a) for a in clf.predict(Xcs[f])]
                    #    init_pred_ind[f] = [i for i,e in enumerate(init_pred[f]) if e==0]
                    #    fold_stances_new[f] = [fold_stances[1][x]['Stance'] for x in init_pred_ind[f]]
                    #    Xcs_new[f],Xhs_new[f],Xbs_new[f],ys_new[f]=generate_features(fold_stances_new[f],d,str(f),model,binary=False)




                    print("Score for fold "+ str(fold) + " was - " + str(score))
                    if score > best_score:
                        best_score = score
                        best_fold1 = clf
                        best_fold2 = clf2


                #Run on Holdout set and report the final score on the holdout set
                predicted = [LABELS[3] if a==1 else LABELS[0] for a in best_fold1.predict(X_baseline_holdout)]
                test_pred = dict()
                test_pred_ind = dict()
                test_pred = [int(a) for a in best_fold1.predict(X_baseline_holdout)]
                test_pred_ind = [i for i,e in enumerate(test_pred) if e==0]
                
                Xc_holdout_new = [Xc_holdout[x] for x in test_pred_ind]
                test_pred_new = [LABELS[int(a)] for a in best_fold2.predict(Xc_holdout_new)]
                final = predicted
                for i,e in enumerate(test_pred_ind):
                    final[e] = test_pred_new[i]

                actual = [LABELS[int(a)] for a in y_holdout]
                print('confusion matrix for randomforestclassifier')
                fscore[pmtune1].append(report_score(actual,final))


    #SVM



