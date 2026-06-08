import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from fintell_package.cleaner import clean_data
from fintell_package.registry import load_model_dl, load_word2vec, load_encoder
from fintell_package.dl_logic.tokenizer import tokenizer
from fintell_package.dl_logic.embedder import transform_embedding


app = FastAPI(
    title="Fintell Deep Learning API",
    description="API de prédiction de sentiment avec modèle Deep Learning Word2Vec + LSTM/GRU/BiGRU.",
    version="0.0.1"
)

model = load_model_dl()
word2vec = load_word2vec()
encoder = load_encoder()


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
    df = pd.DataFrame({"review_text": [review]})

    df_clean = clean_data(df, inference=True)

    if df_clean.empty:
        return {
            "review": review,
            "error": "Review trop courte ou invalide après nettoyage."
        }

    X_tok = tokenizer(df_clean["review_text"])
    X_pad = transform_embedding(X_tok, word2vec)

    proba = model.predict(X_pad)
    pred_id = np.argmax(proba, axis=1)[0]

    label = encoder.inverse_transform([pred_id])[0]

    return {
        "review": review,
        "prediction": label,
        "confidence": float(np.max(proba))
    }


@app.post("/predict")
def predict(input: ReviewInput):
    df = pd.DataFrame({"review_text": input.reviews})

    df_clean = clean_data(df, inference=True)

    if df_clean.empty:
        return {
            "error": "Toutes les reviews sont trop courtes ou invalides après nettoyage."
        }

    X_tok = tokenizer(df_clean["review_text"])
    X_pad = transform_embedding(X_tok, word2vec)

    proba = model.predict(X_pad)
    pred_ids = np.argmax(proba, axis=1)

    labels = encoder.inverse_transform(pred_ids)

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
                np.max(proba, axis=1)
            )
        ]
    }
