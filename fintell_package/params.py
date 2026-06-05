import os
from pathlib import Path

#FILEPATHS
ROOT_DIR            = Path(__file__).resolve().parent.parent
RAW_DIR             = ROOT_DIR / "data" / "raw_data"
PREPROCESSED_DIR    = ROOT_DIR / "data" / "preprocessed_data"
PREPROCESSED_DIR_DL = ROOT_DIR / "data" / "preprocessed_data_dl"
MODEL_DIR           = ROOT_DIR / "models"
MODEL_DIR_ML        = MODEL_DIR / "ml"
MODEL_DIR_DL        = MODEL_DIR / "dl"
MODEL_DIR_DL_PLOTS = MODEL_DIR_DL / "plots"

for d in [RAW_DIR, PREPROCESSED_DIR, PREPROCESSED_DIR_DL, MODEL_DIR_ML, MODEL_DIR_DL, MODEL_DIR_DL_PLOTS]:
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

#FLAGS MODELS
MODEL_NAME = os.getenv("MODEL_NAME", "svc")

# DL HYPERPARAMS
MAXLEN      = int(os.getenv("MAXLEN", "80"))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "60"))
