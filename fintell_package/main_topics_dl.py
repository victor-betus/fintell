import pandas as pd
import numpy as np
from fintell_package.registry import save_word2vec, load_word2vec, save_encoder, load_encoder, save_model_dl, load_model_dl, save_plot, save_run_metadata, save_vocab, load_vocab
from fintell_package.cleaner import clean_data
from fintell_package.dl_logic.tokenizer import tokenizer
from fintell_package.dl_logic.embedder import fit_word2vec, transform_embedding
from fintell_package.dl_logic.encoder import fit_encoder, transform_encoder
from fintell_package.dl_logic.model_topics_dl import init_model, train_model, evaluate_model
from fintell_package.data import save_dl_data, load_dl_data
from fintell_package.params import (
    PREPROCESSED_DIR_TOPIC_DL,
    GCS_PROJECT_ID, GCS_BUCKET_NAME,
    GCS_RAW_TRAIN, GCS_RAW_VAL,
    GCS_PREFIX_TOPIC_DL_DATA, GCS_PREFIX_TOPIC_DL_MODELS, GCS_PREFIX_TOPIC_DL_PLOTS, GCS_PREFIX_TOPIC_DL_RUNS,
    LOAD_TOPIC_DL_PREPROCESSED,
    MAXLEN, VECTOR_SIZE,
    SAMPLE_SIZE,
    PREPROCESS_LEMMATIZE,
    PREPROCESS_STOPWORDS,
    MODEL_TOPIC_DL_NAME,
    USE_CLASS_WEIGHT,
    EMBEDDER_NAME,
    MODEL_DIR_TOPIC_DL, MODEL_DIR_TOPIC_DL_RUNS,
)

from tensorflow.keras.preprocessing.sequence import pad_sequences


def _build_vocab(X_tok):
    vocab = {}
    for tokens in X_tok:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1  # 0 réservé pour padding
    return vocab


def _tokens_to_integers(X_tok, vocab):
    return [[vocab.get(t, 0) for t in tokens] for tokens in X_tok]


def preprocess_topic(split='train'):

    print(f"\n🔄 Preprocessing Topics DL — split: {split}")

    if LOAD_TOPIC_DL_PREPROCESSED:
        print(f"📦 Loading cached topic DL data from GCS...")
        encoder = load_encoder(local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
        X_pad = load_dl_data('X_pad', split, GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
        y_enc = load_dl_data('y_enc', split, GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
        print(f"✅ Cached data loaded — X_pad: {X_pad.shape}, y_enc: {y_enc.shape}")
        return X_pad, y_enc, encoder

    print(f"📥 Loading {split} data from GCS...")
    df = pd.read_parquet(GCS_RAW_TRAIN if split == 'train' else GCS_RAW_VAL)
    df = clean_data(df)
    df = df[df['topic_label_ALL'] != 'Undefined'].copy()
    if SAMPLE_SIZE > 0:
        df = df.sample(SAMPLE_SIZE, random_state=42)
    X = df['review_text']
    y = df['topic_label_ALL']
    print(f"✅ Loaded {len(df)} rows | classes: {y.value_counts().to_dict()}")

    print(f"✂️ Tokenizing...")
    X_tok = tokenizer(X)

    if EMBEDDER_NAME == 'word2vec':
        if split == 'train':
            print(f"🧠 Training Word2Vec...")
            word2vec = fit_word2vec(X_tok)
            save_word2vec(word2vec, local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
        else:
            print(f"📦 Loading Word2Vec...")
            word2vec = load_word2vec(local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)

        print(f"🔢 Embedding + padding (maxlen={MAXLEN})...")
        X_pad = transform_embedding(X_tok, word2vec)

    elif EMBEDDER_NAME == 'keras_embedding':
        if split == 'train':
            print(f"📖 Building vocabulary...")
            vocab = _build_vocab(X_tok)
            save_vocab(vocab, local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
            print(f"✅ Vocab size: {len(vocab)}")
        else:
            print(f"📦 Loading vocabulary...")
            vocab = load_vocab(local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)

        print(f"🔢 Converting to integers + padding (maxlen={MAXLEN})...")
        X_int = _tokens_to_integers(X_tok, vocab)
        X_pad = pad_sequences(X_int, maxlen=MAXLEN, padding='post', truncating='post')

    print(f"✅ X_pad shape: {X_pad.shape}")
    save_dl_data(X_pad, 'X_pad', split, GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)

    if split == 'train':
        print(f"🏷️ Fitting encoder...")
        encoder = fit_encoder(y)
        save_encoder(encoder, local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
    else:
        print(f"📦 Loading encoder...")
        encoder = load_encoder(local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)

    y_enc = transform_encoder(y, encoder)
    print(f"✅ y_enc shape: {y_enc.shape} | classes: {encoder.classes_}")
    save_dl_data(y_enc, 'y_enc', split, GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)

    print(f"✅ Preprocessing done — {split}")
    return X_pad, y_enc, encoder


def train():
    print("\n🤖 Training Topics DL model...")

    print("📥 Loading X_pad and y_enc from GCS...")
    X_train_pad = load_dl_data('X_pad', 'train', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
    y_train_enc = load_dl_data('y_enc', 'train', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
    X_val_pad   = load_dl_data('X_pad', 'val',   GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
    y_val_enc   = load_dl_data('y_enc', 'val',   GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
    print(f"✅ X_train: {X_train_pad.shape} | X_val: {X_val_pad.shape}")

    print("🏗️ Initializing model...")
    encoder = load_encoder(local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
    num_classes = len(encoder.classes_)

    if EMBEDDER_NAME == 'keras_embedding':
        vocab = load_vocab(local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
        vocab_size = len(vocab) + 1  # +1 pour le padding index 0
        model = init_model(MAXLEN, VECTOR_SIZE, MODEL_TOPIC_DL_NAME, num_classes=num_classes, vocab_size=vocab_size)
    else:
        model = init_model(MAXLEN, VECTOR_SIZE, MODEL_TOPIC_DL_NAME, num_classes=num_classes)

    print("🚀 Training...")
    model, history, plot_path = train_model(X_train_pad, y_train_enc, X_val_pad, y_val_enc, model)

    print("💾 Saving model...")
    save_model_dl(model, model_name=MODEL_TOPIC_DL_NAME, local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_MODELS)
    save_plot(plot_path, model_name=MODEL_TOPIC_DL_NAME, gcs_prefix=GCS_PREFIX_TOPIC_DL_PLOTS)
    print("✅ Training done")


def evaluate():
    print("\n📊 Evaluating Topics DL model...")

    print("📥 Loading X_val_pad and y_val_enc from GCS...")
    X_val_pad = load_dl_data('X_pad', 'val', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)
    y_val_enc = load_dl_data('y_enc', 'val', GCS_PROJECT_ID, GCS_BUCKET_NAME, PREPROCESSED_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_DATA)

    print("📦 Loading model...")
    model = load_model_dl(model_name=MODEL_TOPIC_DL_NAME, local_dir=MODEL_DIR_TOPIC_DL, gcs_prefix=GCS_PREFIX_TOPIC_DL_MODELS)

    accuracy, f1, report = evaluate_model(X_val_pad, y_val_enc, model)
    print(f"✅ accuracy: {accuracy:.4f} | f1: {f1:.4f}")
    print(report)

    save_run_metadata(accuracy, f1, report, {
        "maxlen": MAXLEN,
        "vector_size": VECTOR_SIZE,
        "sample_size": SAMPLE_SIZE,
        "lemmatize": PREPROCESS_LEMMATIZE,
        "stopwords": PREPROCESS_STOPWORDS,
        "model_topic_dl_name": MODEL_TOPIC_DL_NAME,
        "use_class_weight": USE_CLASS_WEIGHT,
        "embedder_name": EMBEDDER_NAME,
    }, model_name=MODEL_TOPIC_DL_NAME, local_dir=MODEL_DIR_TOPIC_DL_RUNS, gcs_prefix=GCS_PREFIX_TOPIC_DL_RUNS)


if __name__ == "__main__":
    preprocess_topic('train')
    preprocess_topic('val')
    train()
    evaluate()
