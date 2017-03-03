from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import csv
import scorer.FNCException as FNCException
import numpy as np

def loadData(data_file, field_names):
    data = None
    try:
        with open(data_file) as f_in:
            reader = csv.DictReader(f_in)
            if reader.fieldnames != field_names:
                error = 'ERROR: Incorrect headers in: {}'.format(data_file)
                raise FNCException(error)
            else:
                data = list(reader)

            if data is None:
                error = 'ERROR: No data found in: {}'.format(data_file)
                raise FNCException(error)
    except IOError:
        error = "ERROR: Could not find file: {}".format(data_file)
        raise FNCException(error)

    return data
    
body_fields = ['Body ID','articleBody']
stance_fields = ['Headline','Body ID','Stance']
stance_map = {'agree':1, 'disagree':1, 'discuss':1, 'unrelated':0}

train_bodies = loadData('train_bodies.csv', body_fields)
train_stances = loadData('train_stances.csv', stance_fields)
train_gold = [stance_map[y[stance_fields[2]]] for y in train_stances]

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
lr = LogisticRegression(solver='sag', n_jobs=-1)

#  TODO try this 3 ways:
# - Build counts and tf-idf using only article headlines
# - Build counts and tf-idf using only article bodies
# - Build counts and tf_idf by appending headlines and bodies
train_headlines = [x[stance_fields[0]] for x in train_stances]
train_headline_counts = vectorizer.fit_transform(train_headlines)
train_tfidf = transformer.fit_transform(train_headline_counts)
train_body_counts = vectorizer.transform([x[body_fields[1]] for x in train_bodies])
train_body_tfidf = transformer.transform(train_body_counts)
final_data = [ np.append(row[1], train_body_tfidf[row[0]['Body ID']]) for row in zip(train_stances, train_tfidf)]
lr.fit(train_tfidf, train_gold)

test_bodies = loadData('test_bodies.csv', body_fields)
test_stances = loadData('test_stances.csv', stance_fields)
test_gold = [stance_map[y[stance_fields[2]]] for y in test_stances]
test_headlines = [x[stance_fields[0]] for x in test_stances]
testCounts = vectorizer.transform(test_headlines)
test_tfidf = transformer.transform(testCounts)
lr.score(test_tfidf, test_gold)