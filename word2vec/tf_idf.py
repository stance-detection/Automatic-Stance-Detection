def doc_to_tf(text, ngram=1):
    words = tokenise(text)
    ret = defaultdict(float)
    for i in range(len(words)):
        for j in range(1, ngram+1):
            if i - j < 0:
                break
            word = [words[i-k] for k in range(j)]
            ret[word[0] if ngram == 1 else tuple(word)] += 1.0
    return ret

# Convert a document to GloVe vectors, by computing tf-idf of each word * GLoVe of word / total tf-idf for document
def doc_to_glove(doc):
    doc_tf = doc_to_tf(doc)
    doc_tf_idf = defaultdict(float)
    for word, tf in doc_tf.items():
        doc_tf_idf[word] = tf * idf[word]

    doc_vector = np.zeros(glove_vectors['glove'].shape[0])
    if np.sum(list(doc_tf_idf.values())) == 0.0:  # edge case: document is empty
        return doc_vector

    for word, tf_idf in doc_tf_idf.items():
        if word in glove_vectors:
            doc_vector += glove_vectors[word] * tf_idf
    doc_vector /= np.sum(list(doc_tf_idf.values()))
    return doc_vector

def tf_idf():
    # Build corpus of article bodies and headlines in training dataset
    corpus = np.r_[train_all[:, 1], train_all[:, 0]]  # 0 to 44973 are bodies, 44974 to 89943 are headlines

    # Learn idf of every word in the corpus
    df = defaultdict(float)
    for doc in tqdm(corpus):
        words = tokenise(doc)
        seen = set()
        for word in words:
            if word not in seen:
                df[word] += 1.0
                seen.add(word)

    num_docs = corpus.shape[0]
    idf = defaultdict(float)
    for word, val in tqdm(df.items()):
        idf[word] = np.log((1.0 + num_docs) / (1.0 + val)) + 1.0  # smoothed idf

    # Load GLoVe word vectors
    f_glove = open("data/glove.6B.50d.txt", "rb")  # download from https://nlp.stanford.edu/projects/glove/
    glove_vectors = {}
    for line in tqdm(f_glove):
        glove_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))
    return doc_to_glove()
