import sys
import numpy as np

from lstm.feature_engineering import word_features, gen_or_load_feats
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from lstm.my_lstm import LSTM


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_embedding = gen_or_load_feats(word_features, h, b, "features/embedding."+name+".npy")
    #return X_embedding,y
    X = np.c_[X_embedding]
    print(type(X[0][0]))
    return X,y



if __name__ == "__main__":

    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)

    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:  # iterates over keys
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))

    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        #print(X_train)
        #print(type(X_train))
        #print(X_train.shape)
        X_test = Xs[fold]
        y_test = ys[fold]

        #TODO construct LSTMs - headline and body
        #lstm = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        lstm = LSTM(X_train, y_train)
        predicted = [LABELS[int(a)] for a in lstm.test(X_test, y_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = lstm



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    report_score(actual,predicted)
