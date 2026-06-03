from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.feature_extraction.text import TfidfVectorizer

# Goal : train the vectorized only on the test set
def fit_vectorizer(X):
    tfidf = TfidfVectorizer(max_features=10000)
    tfidf.fit(X)
    return tfidf

# Goal : transform without training
def transform_vectorizer(X, tfidf):
    return tfidf.transform(X)


# def vectorizer(X) :

#     tfidf = TfidfVectorizer(
#         max_features=10000
#     )

#     X_tfidf = tfidf.fit_transform(
#         X
#     )

#     return X_tfidf
