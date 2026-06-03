# generic imports
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

# fintell package imports
from cleaner import clean_data
from stopwords import stopwords
from vectorizer import fit_vectorizer, transform_vectorizer
from smote import oversampling_smote
from lemmatizer import lemmatizer
from model import train_model, evaluate_model, predict_model
from registry import load_model, save_model
from params import MODEL_NAME

def preprocess_sentiment(df, lem=False, stop=True, smot=True, vector=True, tfidf=None, split=''):

    # 1. Clean data
    print(f"🧹 Cleaning data... {len(df)} rows")
    df = clean_data(df)
    print(f"✅ Clean done — {len(df)} rows")

    # 2. Stopwords
    if stop:
        print("🔇 Removing stopwords...")
        df = stopwords(df)
        print("✅ Stopwords done")

    # 3. Lemmatize
    if lem:
        print("🌿 Lemmatizing...")
        df['review_text'] = df['review_text'].apply(lemmatizer)
        print("✅ Lemmatization done")

    # 4. Create X, y
    X = df['review_text']
    y = df['review_sentiment_label']

    # # 5. Split
    # print("✂️ Splitting train/test...")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(f"✅ Split done — train: {len(X_train)}, test: {len(X_test)}")

    # 5. Save
    print("💾 Saving preprocessed data...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../data/preprocessed_data/{split}_reviews_{timestamp}_lemma-{'T' if lem else 'F'}_stop-{'T' if stop else 'F'}_smote-{'T' if smot else 'F'}.parquet"
    df[['review_text', 'review_sentiment_label']].to_parquet(output_path)
    print(f"✅ Saved to {output_path}")

    # 6. Vectorize
    print('🔢 Vectorizing...')
    if vector:
        if tfidf:
            print('🔢 Transforming ...')
            X = transform_vectorizer(X, tfidf)
            print('✅ Vectorization done')
        else:
            print('🔢 Training vectorizer...')
            tfidf = fit_vectorizer(X)
            print('🔢 Transforming ...')
            X = transform_vectorizer(X, tfidf)
            print('✅ Vectorization done')

    # 7. SMOTE
    if smot:
        print("⚖️ Applying SMOTE...")
        X, y = oversampling_smote(X, y, random_state=42, k_neighbors=5)
        print("✅ SMOTE done")


    return X, y, tfidf

def train(X_train, y_train):
    print(f"🤖 Training model ({MODEL_NAME})...")
    model = train_model(X_train, y_train, model_name=MODEL_NAME)
    print("✅ Training done")
    return model

def evaluate(X_test, y_test, model):
    print("📊 Evaluating...")
    accuracy, f1, report = evaluate_model(X_test, y_test, model)
    print("✅ Evaluation done")
    return accuracy, f1, report

def pred(X_test):
    pass



if __name__ == "__main__":

    # TRAIN
    print("\n--- TRAIN ---")
    df_train = pd.read_parquet('../data/raw_data/fintell_train.parquet')
    X, y, tfidf = preprocess_sentiment(df_train, split='train')
    model = train(X, y)
    save_model(model, tfidf, model_name=MODEL_NAME)
    accuracy, f1, report = evaluate(X, y, model)
    print(f"accuracy: {accuracy:.4f} | f1: {f1:.4f}")
    print(report)

    # VAL
    print("\n--- VAL ---")
    df_val = pd.read_parquet('../data/raw_data/fintell_val.parquet')
    X, y, _ = preprocess_sentiment(df_val, tfidf=tfidf, smot=False, split='val')
    accuracy, f1, report = evaluate(X, y, model)
    print(f"accuracy: {accuracy:.4f} | f1: {f1:.4f}")
    print(report)

    # # TEST
    # print("\n--- TEST ---")
    # df_test = pd.read_parquet('../data/raw_data/fintell_test.parquet')
    # model, tfidf = load_model()
    # X, y, _ = preprocess_sentiment(df_test, tfidf=tfidf, smot=False, split='test')
    # accuracy, f1, report = evaluate(X, y, model)
    # print(f"accuracy: {accuracy:.4f} | f1: {f1:.4f}")
    # print(report)
