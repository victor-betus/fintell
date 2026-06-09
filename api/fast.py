import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from fintell_package.cleaner import clean_data
from fintell_package.registry import (
    load_model_dl_prod,
    load_encoder_prod,
    load_tokenizer_prod
)
from fintell_package.dl_logic.main_sentiment import predict_sentiment
from fintell_package.params import (
    GCS_PROD_SENTIMENT_MODEL,
    GCS_PROD_SENTIMENT_ENCODER,
    GCS_PROD_SENTIMENT_TOKENIZER,
    MODEL_DIR_DL
)


app = FastAPI(
    title="Fintell Deep Learning API",
    description="API de prédiction de sentiment avec modèle Deep Learning BiGRU.",
    version="0.1.0"
)


model = load_model_dl_prod(
    GCS_PROD_SENTIMENT_MODEL,
    MODEL_DIR_DL
)

encoder = load_encoder_prod(
    GCS_PROD_SENTIMENT_ENCODER,
    MODEL_DIR_DL
)

tok = load_tokenizer_prod(
    GCS_PROD_SENTIMENT_TOKENIZER,
    MODEL_DIR_DL
)


class ReviewInput(BaseModel):
    reviews: list[str]


@app.get("/")
def root():
    return {
        "api": "FINTELL Deep Learning API",
        "status": "running"
    }


@app.get("/predict_one")
def predict_one(review: str):

    df = pd.DataFrame({
        "review_text": [review]
    })

    df_clean = clean_data(
        df,
        inference=True
    )

    if df_clean.empty:
        return {
            "review": review,
            "error": "Review trop courte ou invalide après nettoyage."
        }

    labels, confidences = predict_sentiment(
        df_clean["review_text"],
        model,
        tok,
        encoder
    )

    return {
        "review": review,
        "prediction": labels[0],
        "confidence": float(confidences[0])
    }


@app.post("/predict")
def predict(input: ReviewInput):

    df = pd.DataFrame({
        "review_text": input.reviews
    })

    df_clean = clean_data(
        df,
        inference=True
    )

    if df_clean.empty:
        return {
            "error": "Toutes les reviews sont trop courtes ou invalides après nettoyage."
        }

    labels, confidences = predict_sentiment(
        df_clean["review_text"],
        model,
        tok,
        encoder
    )

    return {
        "results": [
            {
                "review": review,
                "prediction": label,
                "confidence": float(conf)
            }
            for review, label, conf in zip(
                df_clean["review_text"],
                labels,
                confidences
            )
        ]
    }
