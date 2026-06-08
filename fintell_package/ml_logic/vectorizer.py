from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.feature_extraction.text import TfidfVectorizer

# Goal : train the vectorized only on the test set
def fit_vectorizer(X):
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,1), min_df=5)
    tfidf.fit(X)
    return tfidf

# Goal : transform without training
def transform_vectorizer(X, tfidf):
    return tfidf.transform(X)
