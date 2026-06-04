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
MODEL_NAME = os.environ.get('MODEL_NAME')

