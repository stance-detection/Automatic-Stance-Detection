import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet, TestDataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.submission_writer import write_submission


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/agreeing."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/discussing."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

def generate_test_features(stances,dataset,name):
    h, b, = [],[]

    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X


STAGE1_Y_MAP = {0:0, 1:0, 2:0, 3:1}

def do_test():
    # TRAIN
    d = DataSet()
    train_stances = d.stances
    X_train,y_train = generate_features(train_stances,d,"train") #y_train are ints
    del d


    y_train_stage1 = [STAGE1_Y_MAP[y] for y in y_train]

    stage1_clf = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=14128, verbose=True)
    stage1_clf.fit(X_train, y_train_stage1)


    del y_train_stage1

    X_train_stage2 = [X_train[i] for i in range(len(X_train)) if y_train[i] != 3]
    y_train_stage2 = [y for y in y_train if y != 3]
    #print(y_train_stage2)


    stage2_clf = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=14128, verbose=True)
    stage2_clf.fit(X_train_stage2, y_train_stage2)

    del X_train
    del y_train
    del X_train_stage2
    del y_train_stage2

    # TEST
    test_d = TestDataSet()
    test_stances = test_d.stances
    X_test = generate_test_features(test_stances,test_d,"test") # TODO add embedding features

    stage1_predictions = [int(a) for a in stage1_clf.predict(X_test)]
    #print(stage1_predictions)
    related_ids = [i for i in range(len(stage1_predictions)) if stage1_predictions[i] == 0]

    X_test_stage2 = [X_test[i] for i in related_ids]
    stage2_predictions = [int(a) for a in stage2_clf.predict(X_test_stage2)]
    #print(stage2_predictions)

    final_predictions = []
    for i in range(len(stage1_predictions)):
        if i in related_ids:
            prediction = stage2_predictions[related_ids.index(i)]
            final_predictions.append(prediction)
        else:
            final_predictions.append(3)
    #print(final_predictions)

    write_submission(test_d, final_predictions, 'submission.csv')


def do_reg():
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf_stage1 = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
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


if __name__ == "__main__":

    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)

    test_mode = True

    if test_mode:
        do_test()
    else:
        do_reg()
