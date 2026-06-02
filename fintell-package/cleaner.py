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

    # $CHALLENGIFY_END

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - keeping reviews > 10 words
    - keeping reviews with labels positive, negative, neutral
    - keeping reviews with topics
    - adding colomns
    - removing useless colomns
    - cleaning whitespaces
    - lowercasing
    - removing numbers
    - removing puncuation
    """

    # Count words and keep only reviews > 10 words
    df['word_count'] = df['review_text'].str.split().str.len()
    df = df[df['word_count'] > 10].copy()

    # Keep only reviews with sentiments & topics labels
    df = df[df['review_sentiment_label'].isin(['positive', 'negative', 'neutral'])].copy()
    df = df[df['topic_label_ALL'] != 'Undefined'].copy()


    # Remove useless colomns
    df = df.drop(['thumbs_up',
       'replied', 'reply_text', 'reply_date', 'time_to_reply(h)',
       'app_version_head', 'app_version_detail', 'Consistent',
       'reply_sentiment', 'reply_sentiment_label', 'topic_label_NEG',
       'topic_prob_NEG', 'topic_id_POS', 'topic_label_POS', 'topic_prob_POS', 'topic_id_NEG', 'word_count'
        ], axis=1)

    # Basic cleaning
    df['review_text'] = df.review_text.apply(basic_cleaning)

    return df
