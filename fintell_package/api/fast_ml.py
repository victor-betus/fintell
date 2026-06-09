import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fintell_package.cleaner import clean_data
from fintell_package.registry import load_model

app = FastAPI(
    title="Fintell API",
    description="""
    API de prédiction de sentiment sur des reviews produits.

    Le modèle est un LinearSVC entraîné sur un corpus de reviews en anglais.
    Il prédit si une review est **positive**, **negative** ou **neutral**.

    Le modèle et le vectorizer TF-IDF sont chargés automatiquement depuis GCS au démarrage.
    """,
    version="0.0.1"
)

model, tfidf = load_model()

class ReviewInput(BaseModel):
    reviews: list[str]

    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "This product is absolutely amazing, I love it!",
                    "Terrible quality, broke after one day.",
                    "It works fine, nothing special."
                ]
            }
        }

@app.post("/predict")
def predict(input: ReviewInput):
    """
    Prédit le sentiment d'une liste de reviews.

    - **reviews** : liste de strings, chaque string est une review en anglais

    Chaque review est automatiquement nettoyée avant la prédiction :
    suppression de la ponctuation, des chiffres, mise en minuscules.

    ⚠️ Les reviews de moins de 10 mots sont ignorées et n'apparaîtront pas dans les résultats.

    Retourne pour chaque review le texte original et la prédiction : `positive`, `negative` ou `neutral`.
    """
    df = pd.DataFrame({'review_text': input.reviews})
    df = clean_data(df, inference= True)
    X = tfidf.transform(df['review_text'])
    y_pred = model.predict(X)
    return {"results": [{"review": review, "prediction": pred} for review, pred in zip(input.reviews, y_pred.tolist())]}

@app.get("/predict_one")
def predict_one(review: str):
    """
    Prédit le sentiment d'une seule review.

    - **review** : texte de la review en anglais (query parameter)

    La review est automatiquement nettoyée avant la prédiction :
    suppression de la ponctuation, des chiffres, mise en minuscules.

    ⚠️ La review doit contenir au moins 10 mots, sinon aucune prédiction ne sera retournée.

    Retourne le texte original et la prédiction : `positive`, `negative` ou `neutral`.
    """
    df = pd.DataFrame({'review_text': [review]})
    df = clean_data(df, inference= True)
    X = tfidf.transform(df['review_text'])
    y_pred = model.predict(X)
    return {"review": review, "prediction": y_pred[0]}

@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {
    'greeting': 'Pong Agathe'
    }
