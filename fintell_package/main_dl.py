import pandas as pd
import numpy as np
from fintell_package.registry import save_word2vec, load_word2vec, save_encoder, load_encoder, save_model_dl, load_model_dl, save_plot, save_run_metadata
from fintell_package.cleaner import clean_data
from fintell_package.dl_logic.tokenizer import tokenizer
from fintell_package.dl_logic.embedder import fit_word2vec, transform_embedding
from fintell_package.dl_logic.encoder import fit_encoder, transform_encoder
from fintell_package.dl_logic.model_dl import init_model, train_model, evaluate_model
from fintell_package.data import save_dl_data, load_dl_data
from fintell_package.params import (
    PREPROCESSED_DIR_DL,
    GCS_PROJECT_ID, GCS_BUCKET_NAME,
    GCS_RAW_TRAIN, GCS_RAW_VAL,
    LOAD_DL_PREPROCESSED,
    MAXLEN, VECTOR_SIZE,
    SAMPLE_SIZE,
    PREPROCESS_LEMMATIZE,
    PREPROCESS_STOPWORDS,
    MODEL_DL_NAME,
    USE_CLASS_WEIGHT,
    EMBEDDER_NAME
)


def preprocess_dl(split='train'):

    print(f"\n🔄 Preprocessing DL — split: {split}")

    if LOAD_DL_PREPROCESSED:
        print(f"📦 Loading cached DL data from GCS...")
        word2vec = load_word2vec()
        encoder = load_encoder()
        X_pad = load_dl_data('X_pad', split, GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)
        y_enc = load_dl_data('y_enc', split, GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)
        print(f"✅ Cached data loaded — X_pad: {X_pad.shape}, y_enc: {y_enc.shape}")
        return X_pad, y_enc, word2vec, encoder

    print(f"📥 Loading {split} data from GCS...")
    df = pd.read_parquet(GCS_RAW_TRAIN if split == 'train' else GCS_RAW_VAL)
    df = clean_data(df)
    if SAMPLE_SIZE > 0:
        df = df.sample(SAMPLE_SIZE, random_state=42)
    X = df['review_text']
    y = df['review_sentiment_label']
    print(f"✅ Loaded {len(df)} rows | classes: {y.value_counts().to_dict()}")

    print(f"✂️ Tokenizing...")
    X_tok = tokenizer(X)

    if split == 'train':
        print(f"🧠 Training Word2Vec...")
        word2vec = fit_word2vec(X_tok)
        save_word2vec(word2vec)
    else:
        print(f"📦 Loading Word2Vec...")
        word2vec = load_word2vec()

    print(f"🔢 Embedding + padding (maxlen={MAXLEN})...")
    X_pad = transform_embedding(X_tok, word2vec)
    print(f"✅ X_pad shape: {X_pad.shape}")
    save_dl_data(X_pad, 'X_pad', split, GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)

    if split == 'train':
        print(f"🏷️ Fitting encoder...")
        encoder = fit_encoder(y)
        save_encoder(encoder)
    else:
        print(f"📦 Loading encoder...")
        encoder = load_encoder()

    y_enc = transform_encoder(y, encoder)
    print(f"✅ y_enc shape: {y_enc.shape} | classes: {encoder.classes_}")
    save_dl_data(y_enc, 'y_enc', split, GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)

    print(f"✅ Preprocessing done — {split}")
    return X_pad, y_enc, word2vec, encoder


def train():
    print("\n🤖 Training DL model...")

    print("📥 Loading X_pad and y_enc from GCS...")
    X_train_pad = load_dl_data('X_pad', 'train', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)
    y_train_enc = load_dl_data('y_enc', 'train', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)
    X_val_pad = load_dl_data('X_pad', 'val', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)
    y_val_enc = load_dl_data('y_enc', 'val', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)
    print(f"✅ X_train: {X_train_pad.shape} | X_val: {X_val_pad.shape}")

    print("🏗️ Initializing model...")
    model = init_model(MAXLEN, VECTOR_SIZE, MODEL_DL_NAME)

    print("🚀 Training...")
    model, history, plot_path = train_model(X_train_pad, y_train_enc, X_val_pad, y_val_enc, model)

    print("💾 Saving model...")
    save_model_dl(model)
    save_plot(plot_path)
    print("✅ Training done")


def evaluate():
    print("\n📊 Evaluating DL model...")

    print("📥 Loading X_val_pad and y_val_enc from GCS...")
    X_val_pad = load_dl_data('X_pad', 'val', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)
    y_val_enc = load_dl_data('y_enc', 'val', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_DL)

    print("📦 Loading model...")
    model = load_model_dl()

    accuracy, f1, report = evaluate_model(X_val_pad, y_val_enc, model)
    print(f"✅ accuracy: {accuracy:.4f} | f1: {f1:.4f}")
    print(report)

    save_run_metadata(accuracy, f1, report, {
        "maxlen": MAXLEN,
        "vector_size": VECTOR_SIZE,
        "sample_size": SAMPLE_SIZE,
        "lemmatize": PREPROCESS_LEMMATIZE,
        "stopwords": PREPROCESS_STOPWORDS,
        "model_dl_name": MODEL_DL_NAME,
        "use_class_weight": USE_CLASS_WEIGHT,
        "embedder_name": EMBEDDER_NAME,
    })


if __name__ == "__main__":
    preprocess_dl('train')
    preprocess_dl('val')
    train()
    evaluate()
