import os
from pathlib import Path

#FILEPATHS
ROOT_DIR            = Path(__file__).resolve().parent.parent
RAW_DIR             = ROOT_DIR / "data" / "raw_data"
PREPROCESSED_DIR    = ROOT_DIR / "data" / "preprocessed_data"
MODEL_DIR           = ROOT_DIR / "models"

#FLAGS PREPROCESSING
PREPROCESS_LEMMATIZE = os.getenv("PREPROCESS_LEMMATIZE", "false").lower() == "true"
PREPROCESS_STOPWORDS = os.getenv("PREPROCESS_STOPWORDS", "false").lower() == "true"
PREPROCESS_SMOTE     = os.getenv("PREPROCESS_SMOTE",     "false").lower() == "true"
PREPROCESS_VECTORIZE = os.getenv("PREPROCESS_VECTORIZE", "true").lower()  == "true"

#FLAGS MODELS
MODEL_NAME = os.getenv("MODEL_NAME", "svc")

# FLAGS GCS PREPROCESSED
SAVE_PREPROCESSED = os.getenv("SAVE_PREPROCESSED", "false").lower() == "true"
LOAD_PREPROCESSED = os.getenv("LOAD_PREPROCESSED", "false").lower() == "true"

# TARGET
MODEL_TARGET = os.getenv("MODEL_TARGET", "gcs")  # "local" | "gcs"

# GCS
GCS_PROJECT_ID  = os.getenv("GCS_PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
