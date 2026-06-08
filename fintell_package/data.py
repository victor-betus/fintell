from pathlib import Path
from google.cloud import storage
from datetime import datetime
from fintell_package.run_context import RUN_TIMESTAMP

from fintell_package.params import (
    MODEL_NAME,
    RAW_DIR, PREPROCESSED_DIR,
    PREPROCESS_LEMMATIZE, PREPROCESS_STOPWORDS,
    PREPROCESS_SMOTE, PREPROCESS_VECTORIZE,
)

def upload_preprocessed_file_to_bucket(
    project_id: str,
    bucket_name: str,
    local_file: str,
    split: str
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("💾 Saving preprocessed data...")
    filename = (
        f"{split}_reviews_{timestamp}"
        f"_lemma-{'T' if PREPROCESS_LEMMATIZE else 'F'}"
        f"_stop-{'T' if PREPROCESS_STOPWORDS else 'F'}"
        f"_smote-{'T' if PREPROCESS_SMOTE else 'F'}"
        ".parquet"
    )
    destination_path = f"processed_data/{filename}"
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_file)
    file_size_mb = Path(local_file).stat().st_size / (1024 * 1024)
    print("✅ Upload terminé")
    print(f"📄 Fichier : {Path(local_file).name}")
    print(f"📦 Taille : {file_size_mb:.2f} MB")
    print(f"📍 Stocké dans : gs://{bucket_name}/{destination_path}")


def upload_file_to_bucket(
    project_id: str,
    bucket_name: str,
    local_file: str,
    destination_path: str
) -> None:
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    local_path = Path(local_file)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(str(local_path))
    file_size_mb = local_path.stat().st_size / (1024 * 1024)
    print("✅ Upload terminé")
    print(f"📄 Fichier : {local_path.name}")
    print(f"📦 Taille : {file_size_mb:.2f} MB")
    print(f"📍 Stocké dans : gs://{bucket_name}/{destination_path}")


def download_file_from_bucket(
    project_id: str,
    bucket_name: str,
    source_path: str,
    destination_path: str
) -> None:
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob = bucket.blob(source_path)
    blob.download_to_filename(str(destination))
    file_size_mb = destination.stat().st_size / (1024 * 1024)
    print("✅ Download terminé")
    print(f"📄 Fichier : {destination.name}")
    print(f"📦 Taille : {file_size_mb:.2f} MB")
    print(f"📍 Stocké dans : {destination}")


def get_latest_preprocessed_from_gcs(project_id, bucket_name, split, destination_dir):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blobs = sorted(
        [b for b in bucket.list_blobs(prefix=f"processed_data/{split}_reviews_")],
        key=lambda b: b.time_created
    )
    if not blobs:
        raise FileNotFoundError(f"No preprocessed file found in GCS for split='{split}'")
    latest = blobs[-1]
    local_path = Path(destination_dir) / Path(latest.name).name
    download_file_from_bucket(project_id, bucket_name, latest.name, str(local_path))
    return local_path


def get_latest_dl_data_from_gcs(project_id, bucket_name, prefix, destination_dir):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blobs = sorted(
        [b for b in bucket.list_blobs(prefix=prefix)],
        key=lambda b: b.time_created
    )
    if not blobs:
        raise FileNotFoundError(f"No DL data found in GCS for prefix='{prefix}'")
    latest = blobs[-1]
    local_path = Path(destination_dir) / Path(latest.name).name
    download_file_from_bucket(project_id, bucket_name, latest.name, str(local_path))
    return local_path


def save_dl_data(array, name, split, project_id, bucket_name, destination_dir, gcs_prefix="dl_data"):
    import numpy as np
    filename = f"{name}_{split}_{RUN_TIMESTAMP}.npy"
    local_path = Path(destination_dir) / filename
    np.save(str(local_path), array)
    print(f"✅ {name} saved locally: {filename}")
    upload_file_to_bucket(project_id, bucket_name, str(local_path), f"{gcs_prefix}/{filename}")
    print(f"✅ {name} uploaded to GCS: {gcs_prefix}/{filename}")


def load_dl_data(name, split, project_id, bucket_name, destination_dir, gcs_prefix="dl_data"):
    import numpy as np
    local_path = get_latest_dl_data_from_gcs(project_id, bucket_name, f"{gcs_prefix}/{name}_{split}_", destination_dir)
    return np.load(str(local_path))
