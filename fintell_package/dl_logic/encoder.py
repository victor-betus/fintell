from sklearn.preprocessing import LabelEncoder

# Goal : train the encoder only on train
def fit_encoder(X):
    le = LabelEncoder()
    encoder = le.fit(X)
    return encoder

# Goal : transform without training
def transform_encoder(X, encoder):
    return encoder.transform(X)
