import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
from paraphrases import Paraphrases


_wnl = nltk.WordNetLemmatizer()
p_lex = Paraphrases("lexical")
p_many = Paraphrases("one_to_many")

refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract',
        'notwithstanding',
        'although'
        'incorrect',
        'wrong',
        'lie',
        'fabricate',
        'unsupported',
        'untrue',
        'refute',
        'rebut',
        'disprove',
        'contradict',
        'falsify',
        'deceive',
        'invalid'
]

refuting_phrases = [
        'not true',
        'wrong on both counts'
        'actually incorrect',
        'materially false',
        'false representation',
        'failing to',
        'no longer',
        'simply not',
        'in spite',
        'have doubts',
        'some doubts',
        'certain doubts',
        'question marks'
]

agreeing_words = [
	'agree', 
        'true', 
        'valid', 
        'prove', 
        'demonstrate', 
        'confirm', 
        'corroborate', 
        'substantiate', 
        'support', 
        'validate', 
        'concur', 
        'correspond', 
        'legitimate', 
        'reasonable'
]

discussing_words = [
	'ambiguous', 
        'equivocal', 
        'indeterminate', 
        'uncertain', 
        'neutral', 
        'impartial', 
        'debatable', 
        'unclear'
]

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)



def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X



def refuting_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline_tokens = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline_tokens else 0 for word in refuting_words]
        features.extend([1 if phrase in clean_headline else 0 for phrase in refuting_phrases])
        X.append(features)
    return X


def agreeing_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in agreeing_words]
        X.append(features)
    return X


def discussing_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in discussing_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]


    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):
    
    def count_headline_words(headline):
        return len(set(clean(headline).split(" ")))
        
    def count_headline_words_stops(headline):
        return len(remove_stopwords(set(clean(headline).split(" "))))
    
    def pctg_headline_words_in_body(headline, body):
        count = 0.0
        count_early = 0.0
        for headline_token in set(clean(headline).split(" ")):
            if headline_token in clean(body):
                count += 1
            if headline_token in clean(body)[:255]:
                count_early +=1
        headline_counts = count_headline_words(headline)
        return [count/headline_counts, count_early/headline_counts]
    
    def pctg_headline_words_in_body_stops(headline, body):
        count = 0.0
        count_early = 0.0
        for headline_token in remove_stopwords(set(clean(headline).split(" "))):
            if headline_token in clean(body):
                count += 1
            if headline_token in clean(body)[:255]:
                count_early +=1
        headline_counts = count_headline_words_stops(headline)
        return [count/headline_counts, count_early/headline_counts]
    
    def paraphrase_co_occurence(headline, body):
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            for paraphrase_token in p_lex.get(headline_token):
                if paraphrase_token in clean(body):
                    bin_count += 1
                if paraphrase_token in clean(body)[:255]:
                    bin_count_early += 1
            for paraphrase_token in p_many.get(headline_token):
                if paraphrase_token in clean(body):
                    bin_count += 1
                if paraphrase_token in clean(body)[:255]:
                    bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body)
                 + pctg_headline_words_in_body(headline, body)
                 + pctg_headline_words_in_body_stops(headline, body)
                 + paraphrase_co_occurence(headline, body))


    return X
