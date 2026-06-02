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
from preproc_smote import oversampling_smote
from lemmatizer import lemmatizer

def preprocess_sentiment(df, lem=True, stop=True, smot=False, vector=True):

    # 1. Clean data
    df = clean_data(df)

    # 2. Stopwords
    if stop:
        df = stopwords(df)

    # 3. Lemmatize
    if lem:
        df['review_text'] = df['review_text'].apply(lemmatizer)

    # 4. Create X, y
    X = df['review_text']
    y = df['review_sentiment_label']

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Vectorize
    if vector:
        X_train = vectorizer(X_train)
        X_test = vectorizer(X_test)

    # 7. SMOTE
    if smot:
        X_train, y_train = oversampling_smote(X_train, y_train)

    # 8. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/home/vicb2/code/victor-betus/fintell/data/preprocessed_data/reviews_{timestamp}_lemma-{'T' if lem else 'F'}_stop-{'T' if stop else 'F'}_smote-{'T' if smot else 'F'}.parquet"
    df_processed = pd.DataFrame({'review_text': X, 'label': y})
    df_processed.to_parquet(output_path)

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
