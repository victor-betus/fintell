# generic imports
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

# fintell package imports
from fintell_package.cleaner import clean_data
from fintell_package.stopwords import stopwords
from fintell_package.vectorizer import fit_vectorizer, transform_vectorizer
from fintell_package.smote import oversampling_smote
from fintell_package.lemmatizer import lemmatizer
from fintell_package.model import train_model, evaluate_model, predict_model
from fintell_package.registry import load_model, save_model
from fintell_package.data import upload_preprocessed_file_to_bucket, get_latest_preprocessed_from_gcs
from fintell_package.data import upload_file_to_bucket, download_file_from_bucket

# import params
from fintell_package.params import (
    MODEL_NAME,
    RAW_DIR, PREPROCESSED_DIR,
    PREPROCESS_LEMMATIZE, PREPROCESS_STOPWORDS,
    PREPROCESS_SMOTE, PREPROCESS_VECTORIZE,
    SAVE_PREPROCESSED, LOAD_PREPROCESSED,
    MODEL_TARGET,
    GCS_PROJECT_ID, GCS_BUCKET_NAME,GCS_RAW_TRAIN,GCS_RAW_VAL
)

def preprocess_sentiment(split='train'):

    # 0. Load data from GCS
    print(f"📥 Loading {split} data from GCS...")
    df = pd.read_parquet(GCS_RAW_TRAIN if split == 'train' else GCS_RAW_VAL)
    print(f"✅ Loaded {len(df)} rows")

    # 1. Clean data
    print(f"🧹 Cleaning data...")
    df = clean_data(df)
    print(f"✅ Clean done — {len(df)} rows")

    # 2. Stopwords
    if PREPROCESS_STOPWORDS:
        print("🔇 Removing stopwords...")
        df = stopwords(df)
        print("✅ Stopwords done")

    # 3. Lemmatize
    if PREPROCESS_LEMMATIZE:
        print("🌿 Lemmatizing...")
        df['review_text'] = df['review_text'].apply(lemmatizer)
        print("✅ Lemmatization done")

    # 4. Create X, y
    X = df['review_text']
    y = df['review_sentiment_label']

    # 5. Save locally
    print("💾 Saving preprocessed data...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PREPROCESSED_DIR / f"{split}_reviews_{timestamp}_lemma-{'T' if PREPROCESS_LEMMATIZE else 'F'}_stop-{'T' if PREPROCESS_STOPWORDS else 'F'}.parquet"
    df[['review_text', 'review_sentiment_label']].to_parquet(output_path)
    print(f"✅ Saved to {output_path}")

    # 6. Upload to GCS
    if SAVE_PREPROCESSED:
        upload_preprocessed_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(output_path), split)

    return X, y

def train():
    print(f"\n🤖 Training model ({MODEL_NAME})...")

    # 1. Load preprocessed data from GCS
    local_path = get_latest_preprocessed_from_gcs(
        GCS_PROJECT_ID, GCS_BUCKET_NAME, 'train', PREPROCESSED_DIR
    )
    df = pd.read_parquet(local_path)
    X = df['review_text']
    y = df['review_sentiment_label']
    print(f"✅ Loaded {len(df)} rows from {local_path.name}")

    # 2. Vectorize
    print("🔢 Vectorizing...")
    tfidf = fit_vectorizer(X)
    X = transform_vectorizer(X, tfidf)
    print("✅ Vectorization done")

    # 3. Train
    model = train_model(X, y, model_name=MODEL_NAME)
    print("✅ Training done")

    # 4. Save model + tfidf (local + GCS selon MODEL_TARGET)
    save_model(model, tfidf)

    return model, tfidf

def evaluate():
    print("\n📊 Evaluating...")

    # 1. Load preprocessed val data from GCS
    local_path = get_latest_preprocessed_from_gcs(
        GCS_PROJECT_ID, GCS_BUCKET_NAME, 'val', PREPROCESSED_DIR
    )
    df = pd.read_parquet(local_path)
    X = df['review_text']
    y = df['review_sentiment_label']
    print(f"✅ Loaded {len(df)} rows from {local_path.name}")

    # 2. Load model + tfidf
    model, tfidf = load_model()

    # 3. Vectorize avec le tfidf déjà fitté
    X = transform_vectorizer(X, tfidf)

    # 4. Evaluate
    accuracy, f1, report = evaluate_model(X, y, model)
    print(f"accuracy: {accuracy:.4f} | f1: {f1:.4f}")
    print(report)

    return accuracy, f1, report

def pred(X_test):
    pass

# pas utilisé dans le code, mais utile si le make_run_all

if __name__ == "__main__":
    df_train = pd.read_parquet(GCS_RAW_TRAIN)
    df_val = pd.read_parquet(GCS_RAW_VAL)

    preprocess_sentiment(df_train, split='train')
    preprocess_sentiment(df_val, split='val')
    train()
    evaluate()



    # # TEST
    # print("\n--- TEST ---")
    # df_test = pd.read_parquet('../data/raw_data/fintell_test.parquet')
    # model, tfidf = load_model()
    # X, y, _ = preprocess_sentiment(df_test, tfidf=tfidf, smot=False, split='test')
    # accuracy, f1, report = evaluate(X, y, model)
    # print(f"accuracy: {accuracy:.4f} | f1: {f1:.4f}")
    # print(report)
