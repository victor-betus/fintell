import os
from pathlib import Path

# LOCAL FILEPATHS
ROOT_DIR            = Path(__file__).resolve().parent.parent
RAW_DIR             = ROOT_DIR / "data" / "raw_data"
PREPROCESSED_DIR    = ROOT_DIR / "data" / "preprocessed_data"
PREPROCESSED_DIR_DL = ROOT_DIR / "data" / "preprocessed_data_dl"
MODEL_DIR           = ROOT_DIR / "models"
MODEL_DIR_ML        = MODEL_DIR / "ml"
MODEL_DIR_DL        = MODEL_DIR / "dl"
MODEL_DIR_DL_PLOTS = MODEL_DIR_DL / "plots"
MODEL_DIR_DL_RUNS  = MODEL_DIR_DL / "runs"

for d in [RAW_DIR, PREPROCESSED_DIR, PREPROCESSED_DIR_DL, MODEL_DIR_ML, MODEL_DIR_DL, MODEL_DIR_DL_PLOTS, MODEL_DIR_DL_RUNS]:
    d.mkdir(parents=True, exist_ok=True)

#FLAGS PREPROCESSING
PREPROCESS_LEMMATIZE = os.getenv("PREPROCESS_LEMMATIZE", "false").lower() == "true"
PREPROCESS_STOPWORDS = os.getenv("PREPROCESS_STOPWORDS", "false").lower() == "true"
PREPROCESS_SMOTE     = os.getenv("PREPROCESS_SMOTE",     "false").lower() == "true"
PREPROCESS_VECTORIZE = os.getenv("PREPROCESS_VECTORIZE", "true").lower()  == "true"

# FLAGS GCS PREPROCESSED
SAVE_PREPROCESSED = os.getenv("SAVE_PREPROCESSED", "false").lower() == "true"
LOAD_PREPROCESSED = os.getenv("LOAD_PREPROCESSED", "false").lower() == "true"
LOAD_DL_PREPROCESSED = os.getenv("LOAD_DL_PREPROCESSED", "false").lower() == "true"

# TARGET
MODEL_TARGET = os.getenv("MODEL_TARGET", "gcs")  # "local" | "gcs"

# GCS
GCS_PROJECT_ID  = os.getenv("GCS_PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_RAW_TRAIN = f"gs://{GCS_BUCKET_NAME}/data/raw_data/fintell_train.parquet"
GCS_RAW_VAL   = f"gs://{GCS_BUCKET_NAME}/data/raw_data/fintell_val.parquet"

#FLAGS MODELS
MODEL_NAME = os.getenv("MODEL_NAME", "svc")
MODEL_DL_NAME = os.getenv("MODEL_DL_NAME", "lstm")

# DL HYPERPARAMS
MAXLEN      = int(os.getenv("MAXLEN", "80"))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "60"))
USE_CLASS_WEIGHT = os.getenv("USE_CLASS_WEIGHT", "false").lower() == "true"

# DEV
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "0"))  # 0 = pas de sampling
