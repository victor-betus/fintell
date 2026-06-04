import joblib
from datetime import datetime
from params import MODEL_DIR, MODEL_NAME

def save_model(model, tfidf, model_name=MODEL_NAME):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(model, MODEL_DIR / f"model_{model_name}_{timestamp}.pkl")
    joblib.dump(tfidf, MODEL_DIR / f"tfidf_{model_name}_{timestamp}.pkl")

    print(f"✅ Model saved: model_{model_name}_{timestamp}.pkl")
    print(f"✅ TF-IDF saved: tfidf_{model_name}_{timestamp}.pkl")

def load_model(model_name=MODEL_NAME):
    model_files = sorted(MODEL_DIR.glob(f"model_{model_name}_*.pkl"))
    tfidf_files = sorted(MODEL_DIR.glob(f"tfidf_{model_name}_*.pkl"))

    model = joblib.load(model_files[-1])
    tfidf = joblib.load(tfidf_files[-1])

    print(f"📦 Loading model: {model_files[-1].name}")
    print(f"📦 Loading TF-IDF: {tfidf_files[-1].name}")

    return model, tfidf
