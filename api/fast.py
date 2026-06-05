import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fintell_package.cleaner import clean_data
from fintell_package.registry import load_model

app = FastAPI()
model, tfidf = load_model()

class ReviewInput(BaseModel):
    reviews: list[str]

@app.post("/predict")
def predict(input: ReviewInput):
    df = pd.DataFrame({'review_text': input.reviews})
    df = clean_data(df, inference= True)
    X = tfidf.transform(df['review_text'])
    y_pred = model.predict(X)
    return {"results": [{"review": review, "prediction": pred} for review, pred in zip(input.reviews, y_pred.tolist())]}

@app.get("/predict_one")
def predict_one(review: str):
    df = pd.DataFrame({'review_text': [review]})
    df = clean_data(df, inference= True)
    X = tfidf.transform(df['review_text'])
    y_pred = model.predict(X)
    return {"review": review, "prediction": y_pred[0]}

@app.get("/")
def root():
    return {
    'greeting': 'Pong Agathe'
    }
