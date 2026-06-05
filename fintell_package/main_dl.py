import pandas as pd
import numpy as np
from fintell_package.dl_logic.tokenizer import tokenizer
from fintell_package.dl_logic.embedder import fit_word2vec, transform_embedding
from fintell_package.dl_logic.encoder import fit_encoder, transform_encoder
from fintell_package.dl_logic.model_dl import init_model, train_model, evaluate_model
from fintell_package.params import (
    RAW_DIR, PREPROCESSED_DIR,
    MODEL_DIR_DL,
    GCS_PROJECT_ID, GCS_BUCKET_NAME,
)


# main_dl.py
# ├── preprocess_dl(df, split)
# │   ├── si LOAD_DL_PREPROCESSED=true
# │   │   └── charge X_pad, y_enc, word2vec, encoder depuis GCS → return
# │   ├── si LOAD_DL_PREPROCESSED=false
# │   │   ├── sépare X, y
# │   │   ├── tokenize X → X_tok
# │   │   ├── si split='train' → fit_word2vec + save GCS
# │   │   │   sinon → load word2vec GCS
# │   │   ├── embed + pad → X_pad + save GCS
# │   │   ├── si split='train' → fit_encoder + save GCS
# │   │   │   sinon → load encoder GCS
# │   │   ├── encode y → y_enc + save GCS
# │   │   └── return X_pad, y_enc, word2vec, encoder
# │
# ├── train()
# │   ├── charge X_train_pad, y_train_enc depuis GCS
# │   ├── charge X_val_pad, y_val_enc depuis GCS
# │   ├── init_model(MAXLEN, VECTOR_SIZE)
# │   ├── train_model(...)
# │   └── save model → GCS
# │
# ├── evaluate()
# │   ├── charge X_val_pad, y_val_enc depuis GCS
# │   ├── load model depuis GCS
# │   └── evaluate_model(...)
# │
# └── __main__
#     ├── df_train = pd.read_parquet(...)
#     ├── df_val = pd.read_parquet(...)
#     ├── preprocess_dl(df_train, 'train')
#     ├── preprocess_dl(df_val, 'val')
#     ├── train()
#     └── evaluate()


def preprocess_sentiment(split):

    if LOAD_DL_PREPROCESSED:
        local_path = get_latest_preprocessed_from_gcs(
        GCS_PROJECT_ID, GCS_BUCKET_NAME, split, PREPROCESSED_DIR
        )
        df = pd.read_parquet(local_path)
        X = df['review_text']
        y = df['review_sentiment_label']
        print(f"✅ Loaded {len(df)} rows")
        return X_pad, y_enc, word2vec, encoder

    else:

        # 1. Charge le parquet
        local_path = get_latest_preprocessed_from_gcs(
        GCS_PROJECT_ID, GCS_BUCKET_NAME, split, PREPROCESSED_DIR
        )
        df = pd.read_parquet(local_path)
        X = df['review_text']
        y = df['review_sentiment_label']
        print(f"✅ Loaded {len(df)} rows")
        return X_pad, y_enc, word2vec, encode
        # 2. Tokenize
        X_tok = tokenizer(X)
        # 3. Word2Vec
        # 4. Embed + pad
        # 5. Encoder
        # 6. Encode y
        # 7. Save tout → GCS
        return X_pad, y_enc, word2vec, encoder


    # 2. Tokenize


    # 3. Word2Vec
    if split == 'train':
        word2vec = fit_word2vec(X_tok)
        # TODO: save word2vec → GCS
    else:
        # TODO: load word2vec depuis GCS
        pass

    # 4. Embed + pad
    X_pad = transform_embedding(X_tok, word2vec)
    # TODO: save X_pad → GCS

    # 5. Encoder
    if split == 'train':
        encoder = fit_encoder(y)
        # TODO: save encoder → GCS
    else:
        # TODO: load encoder depuis GCS
        pass

    # 6. Encode y
    y_enc = transform_encoder(y, encoder)
    # TODO: save y_enc → GCS

    return X_pad, y_enc, word2vec, encoder


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
