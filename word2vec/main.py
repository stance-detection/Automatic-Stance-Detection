import sys
import numpy as np

from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import GradientBoostingClassifier
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
import gensim
from gensim.models import word2vec
from scipy.spatial.distance import cosine



def cleantext(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens

def buildWordVector(text, model):
    text = cleantext(text)
    v = np.zeros(300).reshape((1,300))
    count = 0
    for word in text:
        if word in model.vocab:
            v = v + model[word].reshape((1,300))
            count += 1
    if count!= 0:
        v /= count
    
    return v

def generate_features(stances, dataset, name, model):
    h, b, y = [], [], [] 
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
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

    print('calculating features')
    Xc_holdout, Xh_holdout, Xb_holdout, y_holdout = generate_features(hold_out_stances,d,"holdout", model)
    for fold in fold_stances:
        Xcs[fold], Xhs[fold], Xbs[fold], ys[fold] = generate_features(fold_stances[fold],d,str(fold), model)

    best_score = 0
    best_fold = None

    print('training')
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xcs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xcs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        #clf = GradientBoostingClassifier(n_estimators=50, random_state=14128, verbose=False)
        # Try random forest
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    report_score(actual,predicted)



