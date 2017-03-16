import sys
import numpy as np

from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
import gensim
from gensim.models import word2vec
from scipy.spatial.distance import cosine
from sklearn import svm


def cleantext(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
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

def generate_features(stances, dataset, name, model, binary=True):
    h, b, y = [], [], [] 
    for stance in stances:
        if (binary!=True):
            y.append(LABELS.index(stance['Stance']))
        else:
            if LABELS.index(stance['Stance'])<3:
                y.append(0)
            else:
                y.append(1)
        h.append(buildWordVector(stance['Headline'],model))
        b.append(buildWordVector(dataset.articles[stance['Body ID']], model))
    c = np.c_[h,b]

    return c, h, b, y


if __name__ == "__main__":

    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)

    print('Loading word2vec model')
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print('Loading DataSet')
    d = DataSet()
    print('generating folds')
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

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
    print('calculating features')
    Xc_holdout, Xh_holdout, Xb_holdout, y_holdout = generate_features(hold_out_stances,d,"holdout", model, binary=False)
    for fold in fold_stances:
        Xcs[fold], Xhs[fold], Xbs[fold], ys[fold] = generate_features(fold_stances[fold],d,str(fold), model, binary=True)
        _,_,_,ys_true[fold] = generate_features(fold_stances[fold],d,str(fold),model, binary=False)

    binary_flag=dict()
    binary_ind = dict()
    for fold in fold_stances:
        binary_flag[fold] = [1 if x['Stance']=='unrelated' else 0 for x in fold_stances[fold]]
        binary_ind[fold] = [i for i,e in enumerate(binary_flag[fold]) if e==0]

    #Xc_holdout_nb, Xh_holdout_nb, Xb_holdout_nb, y_holdout_nb = generate_features(hold_out_stances,d,"holdout",model,binary=False)
    for fold in fold_stances:
        fold_stances_nb[fold] = [fold_stances[fold][x] for x in binary_ind[fold]]
        Xcs_nb[fold], Xhs_nb[fold], Xbs_nb[fold], ys_nb[fold] = generate_features(fold_stances_nb[fold],d,str(fold), model, binary=False)


    best_score = 0
    best_fold1= None
    best_fold2= None
    del model
    print('training')
    #GradientBoostingClassifier already done.
    #RandomForest Classifier
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xcs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))
        
        X_test = Xcs[fold]
        y_test = ys[fold]
        y_test_true = ys_true[fold]
        #clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        #clf = GradientBoostingClassifier(n_estimators=50, random_state=14128, verbose=False)
        #clf = LogisticRegression()
        clf = RandomForestClassifier(n_estimators=200, n_jobs=4, verbose = True)
        #clf = svm.SVC(kernel = 'rbf', gamma=0.5, C=1, verbose=True)
        clf.fit(X_train, y_train)


        X_train_nb = np.vstack(tuple([Xcs_nb[i] for i in ids]))
        y_train_nb = np.hstack(tuple([ys_nb[i] for i in ids]))
 
        clf2 = RandomForestClassifier(n_estimators=200,n_jobs=4,verbose=True)
        clf2.fit(X_train_nb,y_train_nb)                            
                
        
        predicted = [LABELS[3] if a==1 else LABELS[0] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test_true]

        init_pred = dict()
        init_pred_ind=dict()
        #fold_stances_new=dict()
            
        init_pred[fold] = [int(a) for a in clf.predict(Xcs[fold])]
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
    predicted = [LABELS[3] if a==1 else LABELS[0] for a in best_fold1.predict(Xc_holdout)]
    test_pred = dict()
    test_pred_ind = dict()
    test_pred = [int(a) for a in best_fold1.predict(Xc_holdout)]
    test_pred_ind = [i for i,e in enumerate(test_pred) if e==0]
    
    Xc_holdout_new = [Xc_holdout[x] for x in test_pred_ind]
    test_pred_new = [LABELS[int(a)] for a in best_fold2.predict(Xc_holdout_new)]
    final = predicted
    for i,e in enumerate(test_pred_ind):
        final[e] = test_pred_new[i]

    actual = [LABELS[int(a)] for a in y_holdout]
    print('confusion matrix for randomforestclassifier')
    report_score(actual,final)


    #SVM



