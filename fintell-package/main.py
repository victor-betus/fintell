import pandas as pd
import numpy as np
from cleaner import clean_data
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
# from model import train_model, evaluate_model, predict
from datetime import datetime
from stopwords import stopwords
from vectorizer import vectorizer
from smote import oversampling_smote
from lemmatizer import lemmatizer

def preprocess_sentiment(df, lem=True, stop=True, smot=True, vector=True):

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

    # 5. Split
    print("✂️ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Split done — train: {len(X_train)}, test: {len(X_test)}")

    # 6. Vectorize
    if vector:
        print("🔢 Vectorizing...")
        X_train = vectorizer(X_train)
        X_test = vectorizer(X_test)
        print("✅ Vectorization done")

    # 7. SMOTE
    if smot:
        print("⚖️ Applying SMOTE...")
        X_train, y_train = oversampling_smote(X_train, y_train, random_state=42, k_neighbors=5)
        print("✅ SMOTE done")

    # 8. Save
    print("💾 Saving preprocessed data...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/home/vicb2/code/victor-betus/fintell/data/preprocessed_data/reviews_{timestamp}_lemma-{'T' if lem else 'F'}_stop-{'T' if stop else 'F'}_smote-{'T' if smot else 'F'}.parquet"
    df_processed = pd.DataFrame({'review_text': X, 'label': y})
    df_processed.to_parquet(output_path)
    print(f"✅ Saved to {output_path}")

    return X_train, X_test, y_train, y_test

# def train(X_train, y_train):
#     results = train_model(X_train, y_train)
#     return results

# def evaluate():
#     pass

# def pred():
#     pass

if __name__ == "__main__":
    DATA_PATH = '../data/raw_data/sample_20k.parquet'
    df = pd.read_parquet(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_sentiment(df)
    # train(X_train, y_train)
    # evaluate(X_test, y_test)
