from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizer(X) :

    tfidf = TfidfVectorizer(
        max_features=10000
    )

    X_tfidf = tfidf.fit_transform(
        X
    )

    return X_tfidf
