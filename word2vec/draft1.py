import sys
import numpy as np
import nltk
from tqdm import *
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import gensim
from gensim.models import word2vec
from scipy.spatial.distance import cosine
from sklearn import svm
import pickle

model_dir = 'results/'

def cleantext(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.replace("'",""))

    return tokens

def buildWordVector(text, model):

    text = cleantext(text)
    v = np.zeros(300)
    count = 0
    for word in text:
        if word in model.vocab:
            v = v + model[word]
            count += 1
    if count!= 0:
        v /= count

    return v

def generate_baseline_features(stances, dataset, name, binary=True):
    h, b, y = [],[],[]
    baseline_dir = '../baseline/'
    for stance in (stances):
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

def generate_features(stances, dataset, name, model,binary=True):
    headline, body, y = [], [], [] 
    for stance in tqdm(stances):
        if (binary!=True):
            y.append(LABELS.index(stance['Stance']))
        else:
            if LABELS.index(stance['Stance'])<3:
                y.append(0)
            else:
                y.append(1)

        headline.append(buildWordVector(stance["Headline"],model))
        body.append(buildWordVector(dataset.articles[stance["Body ID"]], model))
    concatenated = np.c_[headline,body]
    return concatenated, headline, body, y


class Features():
    def __init__(self, dataset):
        self.dataset = dataset
        #print('generating folds')
        self.folds, self.hold_out = kfold_split(dataset,n_folds=10)
        self.fold_stances, self.hold_out_stances = get_stances_for_folds(dataset,self.folds,self.hold_out)

        self.ys = dict()
        self.Xcs = dict()
        self.ys_nb = dict()
        self.Xcs_nb = dict()
        self.fold_stances_nb = dict()
        self.ys_true=dict()

        self.Xbasenb = dict()
        self.Xtotalnb=dict()
        self.Xtotal = dict()
        self.X_baseline = dict()
        self.y_baseline = dict()

    def holdoutfeatures(self, model):
        print('calculating features')
        print('holdout set')
        self.Xc_holdout, _, _, self.y_holdout = generate_features(self.hold_out_stances,self.dataset,"holdout", model, binary=False)
        self.X_baseline_holdout, _ = generate_baseline_features(self.hold_out_stances, self.dataset, "holdout", binary= False)
        self.Xtotal_holdout = np.hstack((self.X_baseline_holdout, self.Xc_holdout))

    def foldsfeatures(self, model):
        print('folds')
        for fold in self.fold_stances:
            print("fold=",fold)
            self.Xcs[fold], _ , _ , self.ys_true[fold] = generate_features(self.fold_stances[fold],self.dataset,str(fold), model, binary=False)
            self.ys[fold] = [0 if item <3 else 1 for item in self.ys_true[fold]]
            self.X_baseline[fold], _ = generate_baseline_features(self.fold_stances[fold], self.dataset, str(fold), binary = False)
            self.Xtotal[fold] = np.hstack((self.X_baseline[fold], self.Xcs[fold]))


        binary_flag=dict()
        binary_ind = dict()
        for fold in self.fold_stances:
            binary_flag[fold] = [1 if x['Stance']=='unrelated' else 0 for x in self.fold_stances[fold]]
            binary_ind[fold] = [i for i,e in enumerate(binary_flag[fold]) if e==0]

        for fold in self.fold_stances:
            self.ys_nb[fold] = [self.ys_true[fold][bind] for bind in binary_ind[fold]]
            self.Xcs_nb[fold] = self.Xcs[fold][binary_ind[fold][:],:]
            self.Xbasenb[fold] = self.X_baseline[fold][binary_ind[fold][:],:]
            self.Xtotalnb[fold] = np.hstack((self.Xbasenb[fold], self.Xcs_nb[fold]))


    def consolidated_features_cval(self,cval_ind,fold, ids):
        if cval_ind==0:
            two_stage = False

            X_train = np.vstack(tuple([self.X_baseline[i] for i in ids]))
            y_train = np.hstack(tuple([self.ys_true[i] for i in ids]))

            X_test = self.X_baseline[fold]
            y_test = self.ys[fold]
            y_test_true = self.ys_true[fold]

            X_holdout = self.X_baseline_holdout
            y_holdout = self.y_holdout

            X_stg1 = dict()
            X_stg1["train"] = X_train
            X_stg1["test"] = X_test
            X_stg1["holdout"] = X_holdout
            y_stg1 = dict()
            y_stg1["train"] = y_train
            y_stg1["test"] = y_test
            y_stg1["true"] = y_test_true
            y_stg1["holdout"] = y_holdout

            X_stg2 = dict()
            y_stg2 = dict()

        elif cval_ind==1:
            two_stage = False

            X_train = np.vstack(tuple([self.Xcs[i] for i in ids]))
            y_train = np.hstack(tuple([self.ys_true[i] for i in ids]))

            X_test = self.Xcs[fold]
            y_test = self.ys[fold]
            y_test_true = self.ys_true[fold]

            X_holdout = self.Xc_holdout
            y_holdout = self.y_holdout

            X_stg1 = dict()
            X_stg1["train"] = X_train
            X_stg1["test"] = X_test
            X_stg1["holdout"] = X_holdout
            y_stg1 = dict()
            y_stg1["train"] = y_train
            y_stg1["test"] = y_test
            y_stg1["true"] = y_test_true
            y_stg1["holdout"] = y_holdout

            X_stg2 = dict()
            y_stg2 = dict()

        elif cval_ind==2:

            X_train_stg1 = np.vstack(tuple([self.X_baseline[i] for i in ids]))
            y_train_stg1 = np.hstack(tuple([self.ys[i] for i in ids]))

            X_test_stg1 = self.X_baseline[fold]
            y_test_stg1 = self.ys[fold]
            y_test_true = self.ys_true[fold]

            X_train_stg2 = np.vstack(tuple([self.Xcs_nb[i] for i in ids]))
            y_train_stg2 = np.hstack(tuple([self.ys_nb[i] for i in ids]))

            X_test_stg2 = self.Xcs[fold]

            
            X_holdout_stg1 = self.X_baseline_holdout
            X_holdout_stg2 = self.Xc_holdout
            y_holdout = self.y_holdout

            X_stg1 = dict()
            X_stg1["train"] = X_train_stg1
            X_stg1["test"] = X_test_stg1
            X_stg1["holdout"] = X_holdout_stg1
            y_stg1 = dict()
            y_stg1["train"] = y_train_stg1
            y_stg1["test"] = y_test_stg1
            y_stg1["true"] = y_test_true
            y_stg1["holdout"] = y_holdout

            X_stg2 = dict()
            X_stg2["train"] = X_train_stg2
            X_stg2["test"] = X_test_stg2
            X_stg2["holdout"] = X_holdout_stg2
            y_stg2 = dict()
            y_stg2["train"] = y_train_stg2

        elif cval_ind==3:

            X_train_stg1 = np.vstack(tuple([self.Xtotal[i] for i in ids]))
            y_train_stg1 = np.hstack(tuple([self.ys[i] for i in ids]))

            X_test_stg1 = self.Xtotal[fold]
            y_test_stg1 = self.ys[fold]
            y_test_true = self.ys_true[fold]

            X_train_stg2 = np.vstack(tuple([self.Xtotalnb[i] for i in ids]))
            y_train_stg2 = np.hstack(tuple([self.ys_nb[i] for i in ids]))

            X_test_stg2 = self.Xtotal[fold]
            
            X_holdout_stg1 = self.Xtotal_holdout
            X_holdout_stg2 = self.Xtotal_holdout
            y_holdout = self.y_holdout

            X_stg1 = dict()
            X_stg1["train"] = X_train_stg1
            X_stg1["test"] = X_test_stg1
            X_stg1["holdout"] = X_holdout_stg1
            y_stg1 = dict()
            y_stg1["train"] = y_train_stg1
            y_stg1["test"] = y_test_stg1
            y_stg1["true"] = y_test_true
            y_stg1["holdout"] = y_holdout

            X_stg2 = dict()
            X_stg2["train"] = X_train_stg2
            X_stg2["test"] = X_test_stg2
            X_stg2["holdout"] = X_holdout_stg2
            y_stg2 = dict()
            y_stg2["train"] = y_train_stg2

        elif cval_ind==4:

            X_train_stg1 = np.vstack(tuple([self.Xtotal[i] for i in ids]))
            y_train_stg1 = np.hstack(tuple([self.ys[i] for i in ids]))

            X_test_stg1 = self.Xtotal[fold]
            y_test_stg1 = self.ys[fold]
            y_test_true = self.ys_true[fold]

            X_train_stg2 = np.vstack(tuple([self.Xcs_nb[i] for i in ids]))
            y_train_stg2 = np.hstack(tuple([self.ys_nb[i] for i in ids]))

            X_test_stg2 = self.Xcs[fold]
            
            X_holdout_stg1 = self.Xtotal_holdout
            X_holdout_stg2 = self.Xc_holdout
            y_holdout = self.y_holdout

            X_stg1 = dict()
            X_stg1["train"] = X_train_stg1
            X_stg1["test"] = X_test_stg1
            X_stg1["holdout"] = X_holdout_stg1
            y_stg1 = dict()
            y_stg1["train"] = y_train_stg1
            y_stg1["test"] = y_test_stg1
            y_stg1["true"] = y_test_true
            y_stg1["holdout"] = y_holdout

            X_stg2 = dict()
            X_stg2["train"] = X_train_stg2
            X_stg2["test"] = X_test_stg2
            X_stg2["holdout"] = X_holdout_stg2
            y_stg2 = dict()
            y_stg2["train"] = y_train_stg2
            

        return X_stg1, X_stg2, y_stg1, y_stg2


if __name__ == "__main__":

    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)

    print('Loading DataSet')
    d = DataSet()

    #print('Loading word2vec model')

    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    features = Features(d)
    features.holdoutfeatures(model)
    features.foldsfeatures(model)
    fold_stances = features.fold_stances
    folds = features.folds
    best_score = 0
    best_fold1= None
    best_fold2= None
    del model
    print('training')

    cval = {0:'baseline+ppdb',1:'embedding',2:'baseline+ppdb->embedding',3:'baseline+ppdb+embedding->baseline+ppdb+embedding',4:'baseline+ppdb+embedding->embedding'}
    fscore = []
    two_stage= [ False, False, True, True, True]       
    training_flag = True
    for cval_ind in range(2,5):
        print(cval[cval_ind])
            
        if two_stage[cval_ind] == False:

            if training_flag == True:
                for fold in fold_stances:
                    ids = list(range(len(folds)))
                    del ids[fold]
                    best_score=0

                    X_stg1, _, y_stg1, _ = features.consolidated_features_cval(cval_ind,fold, ids)
                    X_train = X_stg1["train"]
                    X_test = X_stg1["test"]
                    X_holdout = X_stg1["holdout"]

                    y_test_true = y_stg1["true"]
                    y_train = y_stg1["train"]
                    y_test = y_stg1["test"]
                    y_holdout = y_stg1["holdout"]

                    clf = RandomForestClassifier(n_estimators=200, n_jobs=4, verbose=False)
                    clf.fit(X_train, y_train)

                    final = [LABELS[int(a)] for a in clf.predict(X_test)]
                    actual = [LABELS[int(a)] for a in y_test_true]
                                        
                    fold_score, _ = score_submission(actual, final)
                    max_fold_score, _ = score_submission(actual, actual)
                    score = fold_score/max_fold_score


                    print("Score for fold "+ str(fold) + " was - " + str(score))
                    if score > best_score:
                        best_score = score
                        best_fold1 = clf
                        #best_fold2 = clf2
                
                filename = model_dir+ "_" + "_" + cval[cval_ind]
                pickle.dump(best_fold1, open(filename, "wb"))                    
                
                final = [LABELS[int(a)] for a in best_fold1.predict(X_holdout)]
                actual = [LABELS[int(a)] for a in y_holdout]
                report_score(actual,final)
                hold_score, _ = score_submission(actual, final)
                hold_actual, _ = score_submission(actual, actual)
                score = hold_score/ hold_actual
                fscore.append(score)
                del best_fold1, best_score

            else:

                filename = model_dir+ "_" + mode + "_" + cval[cval_ind]
                best_fold1 = pickle.load(open(filename, "rb"))

                final = [LABELS[int(a)] for a in best_fold1.predict(X_holdout)]
                actual = [LABELS[int(a)] for a in y_holdout]
                
                hold_score, _ = score_submission(actual, final)
                hold_actual, _ = score_submission(actual, actual)
                score = hold_score/ hold_actual
                fscore.append(score)

        else:
            if training_flag==True:

                for fold in fold_stances:
                    ids = list(range(len(folds)))
                    del ids[fold]
                    best_score=0

                    X_stg1, X_stg2, y_stg1, y_stg2 = features.consolidated_features_cval(cval_ind,fold, ids)
                    X_train_stg1 = X_stg1["train"]
                    X_test_stg1 = X_stg1["test"]
                    X_holdout_stg1 = X_stg1["holdout"]

                    y_test_true = y_stg1["true"]
                    y_train_stg1 = y_stg1["train"]
                    y_test_stg1 = y_stg1["test"]
                    y_holdout = y_stg1["holdout"]


                    X_train_stg2 = X_stg2["train"]
                    X_test_stg2 = X_stg2["test"] 
                    X_holdout_stg2 = X_stg2["holdout"] 
                    y_train_stg2 = y_stg2["train"] 

                    clf = RandomForestClassifier(n_estimators = 200, n_jobs= 4, verbose=False)
                    clf.fit(X_train_stg1, y_train_stg1)
                    
                    clf2 = RandomForestClassifier(n_estimators= 200, n_jobs= 4, verbose=False)    
                    clf2.fit(X_train_stg2, y_train_stg2)                

                    predicted = [LABELS[3] if a==1 else LABELS[0] for a in clf.predict(X_test_stg1)]
                    actual = [LABELS[int(a)] for a in y_test_true]

                    init_pred = [int(a) for a in clf.predict(X_test_stg1)]
                    init_pred_ind = [i for i,e in enumerate(init_pred) if e==0]

                    Xtest_temp = [X_test_stg2[x] for x in init_pred_ind]
                    predicted_new = [LABELS[int(a)] for a in clf2.predict(Xtest_temp)]
                    final = predicted

                    for i,e in enumerate(init_pred_ind):
                        final[e] = predicted_new[i]

                    fold_score, _ = score_submission(actual, final)
                    max_fold_score, _ = score_submission(actual, actual)
                    score = fold_score/max_fold_score


                    print("Score for fold "+ str(fold) + " was - " + str(score))
                    if score > best_score:
                        best_score = score
                        best_fold1 = clf
                        best_fold2 = clf2

                            
                filename1 = model_dir+ "_" +  "_" + cval[cval_ind] + "stg1"
                pickle.dump(best_fold1, open(filename1, "wb"))                    
                filename2 = model_dir+ "_" + "_" + cval[cval_ind] + "stg2"
                pickle.dump(best_fold2, open(filename2, "wb"))
                    
                predicted = [LABELS[3] if a==1 else LABELS[0] for a in best_fold1.predict(X_holdout_stg1)]
                actual = [LABELS[int(a)] for a in y_holdout]

                test_pred = [int(a) for a in best_fold1.predict(X_holdout_stg1)]
                test_pred_ind = [i for i,e in enumerate(test_pred) if e==0]
                
                Xholdout_temp = [X_holdout_stg2[x] for x in test_pred_ind]
                test_pred_new = [LABELS[int(a)] for a in best_fold2.predict(Xholdout_temp)]
                final = predicted

                for i,e in enumerate(test_pred_ind):
                    final[e] = test_pred_new[i]

                print('confusion matrix for randomforestclassifier')
                report_score(actual,final)
                holdout_score, _ = score_submission(actual, final)
                holdout_max_score, _ = score_submission(actual, actual)
                totalscore = holdout_score/ holdout_max_score
                fscore.append(totalscore)
                del best_fold1, best_fold2, best_score

            else:

                filename1 = model_dir+ "_" + "_" + cval[cval_ind] + "stg1"
                best_fold1 = pickle.load(open(filename1, "rb"))                    
                filename2 = model_dir+ "_" + "_" + cval[cval_ind] + "stg2"
                best_fold2 = pickle.load(open(filename2, "rb"))
                    
                predicted = [LABELS[3] if a==1 else LABELS[0] for a in best_fold1.predict(X_holdout_stg1)]
                actual = [LABELS[int(a)] for a in y_holdout]

                test_pred = [int(a) for a in best_fold1.predict(X_holdout_stg1)]
                test_pred_ind = [i for i,e in enumerate(test_pred) if e==0]
                
                Xholdout_temp = [X_holdout_stg2[x] for x in test_pred_ind]
                test_pred_new = [LABELS[int(a)] for a in best_fold2.predict(Xholdout_temp)]
                final = predicted

                for i,e in enumerate(test_pred_ind):
                    final[e] = test_pred_new[i]

                print('confusion matrix for randomforestclassifier')
                report_score(actual,final)
                holdout_score, _ = score_submission(actual, final)
                holdout_max_score, _ = score_submission(actual, actual)
                totalscore = holdout_score/ holdout_max_score
                fscore.append(totalscore)

    filename = model_dir + "_results" 
    pickle.dump(fscore, open(filename, "wb"))





