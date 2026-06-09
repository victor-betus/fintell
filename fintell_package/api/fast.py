import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fintell_package.cleaner import clean_data
from fintell_package.registry import (
    load_model_dl_prod,
    load_encoder_prod,
    load_tokenizer_prod,
)
from fintell_package.dl_logic.main_sentiment import predict_sentiment
from fintell_package.dl_logic.main_topics import predict_topic
from fintell_package.params import (
    MAXLEN,
    GCS_PROD_SENTIMENT_MODEL,
    GCS_PROD_SENTIMENT_ENCODER,
    GCS_PROD_SENTIMENT_TOKENIZER,
    GCS_PROD_TOPIC_MODEL,
    GCS_PROD_TOPIC_ENCODER,
    GCS_PROD_TOPIC_VOCAB,
    MODEL_DIR_DL,
    MODEL_DIR_TOPIC_DL,
)


app = FastAPI(
    title="FINTELL API",
    description="API de prédiction de sentiment et de catégorie.",
    version="1.0.0",
)


model_sentiment = load_model_dl_prod(GCS_PROD_SENTIMENT_MODEL, MODEL_DIR_DL)
encoder_sentiment = load_encoder_prod(GCS_PROD_SENTIMENT_ENCODER, MODEL_DIR_DL)
tok_sentiment = load_tokenizer_prod(GCS_PROD_SENTIMENT_TOKENIZER, MODEL_DIR_DL)

model_category = load_model_dl_prod(GCS_PROD_TOPIC_MODEL, MODEL_DIR_TOPIC_DL)
encoder_category = load_encoder_prod(GCS_PROD_TOPIC_ENCODER, MODEL_DIR_TOPIC_DL)
vocab_category = load_tokenizer_prod(GCS_PROD_TOPIC_VOCAB, MODEL_DIR_TOPIC_DL)


class ReviewInput(BaseModel):
    reviews: list[str]


def texts_to_sequences_with_vocab(texts, vocab):
    sequences = []

    for text in texts.tolist():
        tokens = str(text).lower().split()

        sequence = [
            vocab.get(token, 0)
            for token in tokens
        ]

        sequences.append(sequence)

    return sequences


def predict_category_with_vocab(texts, model, vocab, encoder):
    sequences = texts_to_sequences_with_vocab(texts, vocab)

    X_pad = pad_sequences(
        sequences,
        maxlen=MAXLEN,
        padding="post",
        truncating="post"
    )

    proba = model.predict(X_pad)

    labels = encoder.inverse_transform(
        np.argmax(proba, axis=1)
    )

    confidences = np.max(proba, axis=1)

    return labels, confidences


@app.get("/")
def root():
    return {
        "api": "FINTELL API",
        "status": "running",
        "endpoints": [
            "GET /predict_sentiment",
            "POST /predict_sentiment",
            "GET /predict_category",
            "POST /predict_category",
        ],
    }


@app.get("/predict_sentiment")
def predict_one_sentiment(review: str):
    df = pd.DataFrame({"review_text": [review]})
    df_clean = clean_data(df, inference=True)

    if df_clean.empty:
        return {
            "review": review,
            "error": "Review trop courte ou invalide après nettoyage.",
        }

    labels, confidences = predict_sentiment(
        df_clean["review_text"],
        model_sentiment,
        tok_sentiment,
        encoder_sentiment,
    )

    return {
        "review": review,
        "sentiment": labels[0],
        "confidence": float(confidences[0]),
    }


@app.post("/predict_sentiment")
def predict_many_sentiments(input: ReviewInput):
    df = pd.DataFrame({"review_text": input.reviews})
    df_clean = clean_data(df, inference=True)

    if df_clean.empty:
        return {
            "error": "Toutes les reviews sont invalides après nettoyage."
        }

    labels, confidences = predict_sentiment(
        df_clean["review_text"],
        model_sentiment,
        tok_sentiment,
        encoder_sentiment,
    )

    return {
        "results": [
            {
                "review": review,
                "sentiment": label,
                "confidence": float(conf),
            }
            for review, label, conf in zip(
                df_clean["review_text"],
                labels,
                confidences,
            )
        ]
    }


@app.get("/predict_category")
def predict_one_category(review: str):
    df = pd.DataFrame({"review_text": [review]})
    df_clean = clean_data(df, inference=True)

    if df_clean.empty:
        return {
            "review": review,
            "error": "Review trop courte ou invalide après nettoyage.",
        }

    labels, confidences = predict_category_with_vocab(
        df_clean["review_text"],
        model_category,
        vocab_category,
        encoder_category,
    )

    return {
        "review": review,
        "category": labels[0],
        "confidence": float(confidences[0]),
    }


@app.post("/predict_category")
def predict_many_categories(input: ReviewInput):
    df = pd.DataFrame({"review_text": input.reviews})
    df_clean = clean_data(df, inference=True)

    if df_clean.empty:
        return {
            "error": "Toutes les reviews sont invalides après nettoyage."
        }

    labels, confidences = predict_category_with_vocab(
        df_clean["review_text"],
        model_category,
        vocab_category,
        encoder_category,
    )

    return {
        "results": [
            {
                "review": review,
                "category": label,
                "confidence": float(conf),
            }
            for review, label, conf in zip(
                df_clean["review_text"],
                labels,
                confidences,
            )
        ]
    }
