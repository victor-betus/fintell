import joblib
from pathlib import Path
from datetime import datetime
from gensim.models import Word2Vec
from google.cloud import storage
from fintell_package.data import upload_file_to_bucket, download_file_from_bucket, get_latest_dl_data_from_gcs
from fintell_package.params import MODEL_DIR, MODEL_DIR_DL, MODEL_NAME, GCS_PROJECT_ID, GCS_BUCKET_NAME, MODEL_TARGET

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


# ─── DL ───────────────────────────────────────────

def save_word2vec(word2vec):
    """Save Word2Vec model locally and upload to GCS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"word2vec_{timestamp}.model"
    local_path = MODEL_DIR_DL / filename
    word2vec.save(str(local_path))
    print(f"✅ Word2Vec saved locally: {filename}")
    if MODEL_TARGET == "gcs":
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_path), f"dl_data/{filename}")
        print(f"✅ Word2Vec uploaded to GCS: dl_data/{filename}")


def load_word2vec():
    """Load the latest Word2Vec model from GCS or local."""
    if MODEL_TARGET == "gcs":
        local_path = get_latest_dl_data_from_gcs(GCS_PROJECT_ID, GCS_BUCKET_NAME, "dl_data/word2vec_", MODEL_DIR_DL)
    else:
        local_path = sorted(MODEL_DIR_DL.glob("word2vec_*.model"))[-1]
    word2vec = Word2Vec.load(str(local_path))
    print(f"📦 Word2Vec loaded: {local_path.name}")
    return word2vec


def save_encoder(encoder):
    """Save LabelEncoder locally and upload to GCS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"encoder_{timestamp}.pkl"
    local_path = MODEL_DIR_DL / filename
    joblib.dump(encoder, local_path)
    print(f"✅ Encoder saved locally: {filename}")
    if MODEL_TARGET == "gcs":
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_path), f"dl_data/{filename}")
        print(f"✅ Encoder uploaded to GCS: dl_data/{filename}")


def load_encoder():
    """Load the latest LabelEncoder from GCS or local."""
    if MODEL_TARGET == "gcs":
        local_path = get_latest_dl_data_from_gcs(GCS_PROJECT_ID, GCS_BUCKET_NAME, "dl_data/encoder_", MODEL_DIR_DL)
    else:
        local_path = sorted(MODEL_DIR_DL.glob("encoder_*.pkl"))[-1]
    encoder = joblib.load(local_path)
    print(f"📦 Encoder loaded: {local_path.name}")
    return encoder

def save_model_dl(model):
    """Save Keras LSTM model locally and upload to GCS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_dl_{timestamp}.keras"
    local_path = MODEL_DIR_DL / filename
    model.save(str(local_path))
    print(f"✅ DL Model saved locally: {filename}")
    if MODEL_TARGET == "gcs":
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_path), f"dl_models/{filename}")
        print(f"✅ DL Model uploaded to GCS: dl_models/{filename}")


def load_model_dl():
    """Load the latest Keras LSTM model from GCS or local."""
    if MODEL_TARGET == "gcs":
        local_path = get_latest_dl_data_from_gcs(GCS_PROJECT_ID, GCS_BUCKET_NAME, "dl_models/model_dl_", MODEL_DIR_DL)
    else:
        local_path = sorted(MODEL_DIR_DL.glob("model_dl_*.keras"))[-1]
    from tensorflow.keras.models import load_model
    model = load_model(str(local_path))
    print(f"📦 DL Model loaded: {local_path.name}")
    return model
