import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
from gensim.models import KeyedVectors


_wnl = nltk.WordNetLemmatizer()
w2v = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s)).lower()

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def get_word_vectors_by_ids(s):
    word_vectors = []
    for word_id in s:
        word_vectors.append(w2v.word_vec(w2v.index2word[word_id]))
    return word_vectors

def convert_to_word_vectors(feats):
    X = []
    for entry in feats:
        features = []
        features.append(get_word_vectors_by_ids(entry[0]))
        features.append(get_word_vectors_by_ids(entry[1]))
        X.append(features)
    return X

def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    feats = np.load(feature_file)
    return convert_to_word_vectors(feats)

def get_word_ids(s):
    word_ids = []
    for word in s:
        if word in w2v.vocab:
            word_ids.append(w2v.vocab[word].index)
    return word_ids


def word_features(headlines, bodies):
    X = [] # [[[headline, words],[body, words]], [[next, headline],[next, body]], ...]
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        features = []
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features.append(get_word_ids(clean_headline))
        features.append(get_word_ids(clean_body))
        #print(features)
        X.append(features)
    return X
