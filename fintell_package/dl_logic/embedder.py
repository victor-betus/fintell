import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Goal: train Word2Vec on tokenized sentences, return model
def fit_word2vec(X):
    word2vec = Word2Vec(sentences=X, vector_size=60, min_count=5, window=10)
    print(f"Vocab size: {len(word2vec.wv)}")
    return word2vec

# Goal: convert a single tokenized sentence into a matrix (n_words x vector_size)
def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec.wv:
            embedded_sentence.append(word2vec.wv[word])

    return np.array(embedded_sentence)

# Goal: convert a list of tokenized sentences into a list of matrices
def embedding(word2vec, sentences):
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed

# Goal: embed + pad all sentences, return fixed-shape matrix (n_reviews x maxlen x vector_size)
def transform_embedding(X, word2vec):
    X_embed = embedding(word2vec, X)   # étape 4
    X_pad = pad_sequences(X_embed, dtype='float32', padding='post', maxlen=80) # étape 5
    return X_pad
