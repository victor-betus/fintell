import json
import joblib
from pathlib import Path
from datetime import datetime
from gensim.models import Word2Vec
from google.cloud import storage
from fintell_package.run_context import RUN_TIMESTAMP
from fintell_package.data import upload_file_to_bucket, download_file_from_bucket, get_latest_dl_data_from_gcs
from fintell_package.params import (
    MODEL_DL_NAME, MODEL_TOPIC_DL_NAME,
    MODEL_DIR, MODEL_DIR_DL, MODEL_DIR_DL_RUNS,
    MODEL_DIR_TOPIC_DL, MODEL_DIR_TOPIC_DL_RUNS,
    MODEL_NAME, GCS_PROJECT_ID, GCS_BUCKET_NAME, MODEL_TARGET,
    GCS_PREFIX_DL_DATA, GCS_PREFIX_DL_MODELS, GCS_PREFIX_DL_PLOTS, GCS_PREFIX_DL_RUNS,
)

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

def save_word2vec(word2vec, local_dir=None, gcs_prefix=GCS_PREFIX_DL_DATA):
    if local_dir is None:
        local_dir = MODEL_DIR_DL
    filename = f"word2vec_{RUN_TIMESTAMP}.model"
    local_path = local_dir / filename
    word2vec.save(str(local_path))
    print(f"✅ Word2Vec saved locally: {filename}")
    if MODEL_TARGET == "gcs":
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_path), f"{gcs_prefix}/{filename}")
        print(f"✅ Word2Vec uploaded to GCS: {gcs_prefix}/{filename}")


def load_word2vec(local_dir=None, gcs_prefix=GCS_PREFIX_DL_DATA):
    if local_dir is None:
        local_dir = MODEL_DIR_DL
    if MODEL_TARGET == "gcs":
        local_path = get_latest_dl_data_from_gcs(GCS_PROJECT_ID, GCS_BUCKET_NAME, f"{gcs_prefix}/word2vec_", local_dir)
    else:
        local_path = sorted(local_dir.glob("word2vec_*.model"))[-1]
    word2vec = Word2Vec.load(str(local_path))
    print(f"📦 Word2Vec loaded: {local_path.name}")
    return word2vec


def save_encoder(encoder, local_dir=None, gcs_prefix=GCS_PREFIX_DL_DATA):
    if local_dir is None:
        local_dir = MODEL_DIR_DL
    filename = f"encoder_{RUN_TIMESTAMP}.pkl"
    local_path = local_dir / filename
    joblib.dump(encoder, local_path)
    print(f"✅ Encoder saved locally: {filename}")
    if MODEL_TARGET == "gcs":
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_path), f"{gcs_prefix}/{filename}")
        print(f"✅ Encoder uploaded to GCS: {gcs_prefix}/{filename}")


def load_encoder(local_dir=None, gcs_prefix=GCS_PREFIX_DL_DATA):
    if local_dir is None:
        local_dir = MODEL_DIR_DL
    if MODEL_TARGET == "gcs":
        local_path = get_latest_dl_data_from_gcs(GCS_PROJECT_ID, GCS_BUCKET_NAME, f"{gcs_prefix}/encoder_", local_dir)
    else:
        local_path = sorted(local_dir.glob("encoder_*.pkl"))[-1]
    encoder = joblib.load(local_path)
    print(f"📦 Encoder loaded: {local_path.name}")
    return encoder


def save_model_dl(model, model_name=None, local_dir=None, gcs_prefix=GCS_PREFIX_DL_MODELS):
    if model_name is None:
        model_name = MODEL_DL_NAME
    if local_dir is None:
        local_dir = MODEL_DIR_DL
    filename = f"model_dl_{model_name}_{RUN_TIMESTAMP}.keras"
    local_path = local_dir / filename
    model.save(str(local_path))
    print(f"✅ DL Model saved locally: {filename}")
    if MODEL_TARGET == "gcs":
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_path), f"{gcs_prefix}/{filename}")
        print(f"✅ DL Model uploaded to GCS: {gcs_prefix}/{filename}")


def load_model_dl(model_name=None, local_dir=None, gcs_prefix=GCS_PREFIX_DL_MODELS):
    if model_name is None:
        model_name = MODEL_DL_NAME
    if local_dir is None:
        local_dir = MODEL_DIR_DL
    if MODEL_TARGET == "gcs":
        local_path = get_latest_dl_data_from_gcs(GCS_PROJECT_ID, GCS_BUCKET_NAME, f"{gcs_prefix}/model_dl_{model_name}_", local_dir)
    else:
        local_path = sorted(local_dir.glob(f"model_dl_{model_name}_*.keras"))[-1]
    from tensorflow.keras.models import load_model
    model = load_model(str(local_path))
    print(f"📦 DL Model loaded: {local_path.name}")
    return model


def save_plot(plot_path, model_name=None, gcs_prefix=GCS_PREFIX_DL_PLOTS):
    if model_name is None:
        model_name = MODEL_DL_NAME
    if MODEL_TARGET == "gcs":
        filename = f"training_history_{model_name}_{RUN_TIMESTAMP}.png"
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(plot_path), f"{gcs_prefix}/{filename}")
        print(f"✅ Plot uploaded to GCS: {gcs_prefix}/{filename}")


def save_run_metadata(accuracy, f1, report, params: dict, model_name=None, local_dir=None, gcs_prefix=GCS_PREFIX_DL_RUNS):
    if model_name is None:
        model_name = MODEL_DL_NAME
    if local_dir is None:
        local_dir = MODEL_DIR_DL_RUNS
    metadata = {
        "timestamp": RUN_TIMESTAMP,
        "params": params,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1, 4),
            "classification_report": report
        }
    }
    filename = f"run_{model_name}_{RUN_TIMESTAMP}.json"
    local_path = local_dir / filename
    with open(local_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    if MODEL_TARGET == "gcs":
        upload_file_to_bucket(GCS_PROJECT_ID, GCS_BUCKET_NAME, str(local_path), f"{gcs_prefix}/{filename}")
        print(f"✅ Run metadata uploaded to GCS: {gcs_prefix}/{filename}")
