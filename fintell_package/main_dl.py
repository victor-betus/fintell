

def train():
    # 1. Load data
    df = pd.read_parquet(...)
    X, y = df['review_text'], df['review_sentiment_label']

    # 2. Preprocess
    X_tok = tokenizer(X)
    word2vec = fit_word2vec(X_tok)
    X_pad = transform_embedding(X_tok, word2vec)
    encoder = fit_encoder(y)
    y_enc = transform_encoder(y, encoder)

    # 3. Train
    model = init_model(maxlen=80, vector_size=60)
    model = train_model(X_pad, y_enc, model)

    return model, word2vec, encoder
