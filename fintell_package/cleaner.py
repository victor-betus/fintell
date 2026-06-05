import pandas as pd
import numpy as np
import string

def basic_cleaning(sentence):
    # $CHALLENGIFY_BEGIN

    # Removing whitespaces
    sentence = sentence.strip()
    # Lowercasing
    sentence = sentence.lower()
    # Removing numbers
    sentence = ''.join(char for char in sentence if not char.isdigit())
    # Removing punctuation
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    return sentence

    # Removing smileys ?

    # $CHALLENGIFY_END

def clean_data(df: pd.DataFrame, inference=False) -> pd.DataFrame:

    if not inference:
        df = df[df['review_sentiment_label'].isin(['positive', 'negative', 'neutral'])].copy()
        df = df[df['topic_label_ALL'] != 'Undefined'].copy()

    # Count words and keep only reviews > 10 words
    df['word_count'] = df['review_text'].str.split().str.len()
    df = df[df['word_count'] > 10].copy()

    if not inference:
        # Remove useless columns
        df = df[['review_text', 'topic_label_ALL', 'review_sentiment_label']]

    # Basic cleaning
    df['review_text'] = df.review_text.apply(basic_cleaning)

    return df
