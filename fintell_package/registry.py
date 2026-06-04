import joblib
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from data import upload_file_to_bucket, download_file_from_bucket
from params import MODEL_DIR, MODEL_NAME, GCS_PROJECT_ID, GCS_BUCKET_NAME, MODEL_TARGET

def save_model(model, tfidf, model_name=MODEL_NAME):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"model_{model_name}_{timestamp}.pkl"
    tfidf_filename = f"tfidf_{model_name}_{timestamp}.pkl"

    local_model = MODEL_DIR / model_filename
    local_tfidf = MODEL_DIR / tfidf_filename

    joblib.dump(model, local_model)
    joblib.dump(tfidf, local_tfidf)

    print(f"✅ Model saved locally: {model_filename}")
    print(f"✅ TF-IDF saved locally: {tfidf_filename}")

    if MODEL_TARGET == "gcs":
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_model), f"models/{model_filename}")
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_tfidf), f"models/{tfidf_filename}")
        print(f"✅ Model uploaded to GCS: models/{model_filename}")
        print(f"✅ TF-IDF uploaded to GCS: models/{tfidf_filename}")

def load_model(model_name=MODEL_NAME):

    if MODEL_TARGET == "gcs":
        client = storage.Client(project=GCS_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET_NAME)

        all_blobs = list(bucket.list_blobs(prefix="models/"))
        model_blobs = sorted([b for b in all_blobs if f"model_{model_name}_" in b.name], key=lambda b: b.name)
        tfidf_blobs = sorted([b for b in all_blobs if f"tfidf_{model_name}_" in b.name], key=lambda b: b.name)

        local_model = MODEL_DIR / Path(model_blobs[-1].name).name
        local_tfidf = MODEL_DIR / Path(tfidf_blobs[-1].name).name

        download_file_from_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, model_blobs[-1].name, str(local_model))
        download_file_from_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, tfidf_blobs[-1].name, str(local_tfidf))

    else:
        local_model = sorted(MODEL_DIR.glob(f"model_{model_name}_*.pkl"))[-1]
        local_tfidf = sorted(MODEL_DIR.glob(f"tfidf_{model_name}_*.pkl"))[-1]

    model = joblib.load(local_model)
    tfidf = joblib.load(local_tfidf)

    print(f"📦 Model loaded: {local_model.name}")
    print(f"📦 TF-IDF loaded: {local_tfidf.name}")

    return model, tfidf
