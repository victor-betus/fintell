import os
from pathlib import Path

# LOCAL FILEPATHS
ROOT_DIR            = Path(__file__).resolve().parent.parent
RAW_DIR             = ROOT_DIR / "data" / "raw_data"
PREPROCESSED_DIR    = ROOT_DIR / "data" / "preprocessed_data"
PREPROCESSED_DIR_DL       = ROOT_DIR / "data" / "preprocessed_data_dl"
PREPROCESSED_DIR_TOPIC_DL = ROOT_DIR / "data" / "preprocessed_data_topic_dl"
MODEL_DIR                 = ROOT_DIR / "models"
MODEL_DIR_ML              = MODEL_DIR / "ml"
MODEL_DIR_DL              = MODEL_DIR / "dl"
MODEL_DIR_DL_PLOTS        = MODEL_DIR_DL / "plots"
MODEL_DIR_DL_RUNS         = MODEL_DIR_DL / "runs"
MODEL_DIR_TOPIC_DL        = MODEL_DIR / "topic_dl"
MODEL_DIR_TOPIC_DL_PLOTS  = MODEL_DIR_TOPIC_DL / "plots"
MODEL_DIR_TOPIC_DL_RUNS   = MODEL_DIR_TOPIC_DL / "runs"

for d in [RAW_DIR, PREPROCESSED_DIR, PREPROCESSED_DIR_DL, PREPROCESSED_DIR_TOPIC_DL,
          MODEL_DIR_ML, MODEL_DIR_DL, MODEL_DIR_DL_PLOTS, MODEL_DIR_DL_RUNS,
          MODEL_DIR_TOPIC_DL, MODEL_DIR_TOPIC_DL_PLOTS, MODEL_DIR_TOPIC_DL_RUNS]:
    d.mkdir(parents=True, exist_ok=True)

# FLAGS PREPROCESSING — true | false
PREPROCESS_LEMMATIZE = os.getenv("PREPROCESS_LEMMATIZE", "false").lower() == "true"
PREPROCESS_STOPWORDS = os.getenv("PREPROCESS_STOPWORDS", "false").lower() == "true"
PREPROCESS_SMOTE     = os.getenv("PREPROCESS_SMOTE",     "false").lower() == "true"
PREPROCESS_VECTORIZE = os.getenv("PREPROCESS_VECTORIZE", "true").lower()  == "true"

# FLAGS GCS CACHE — true = skip preprocessing, load from GCS | false = reprocess
SAVE_PREPROCESSED        = os.getenv("SAVE_PREPROCESSED",        "false").lower() == "true"
LOAD_PREPROCESSED        = os.getenv("LOAD_PREPROCESSED",        "false").lower() == "true"
LOAD_DL_PREPROCESSED     = os.getenv("LOAD_DL_PREPROCESSED",     "false").lower() == "true"
LOAD_TOPIC_DL_PREPROCESSED = os.getenv("LOAD_TOPIC_DL_PREPROCESSED", "false").lower() == "true"

# TARGET — local | gcs
MODEL_TARGET = os.getenv("MODEL_TARGET", "gcs")

# GCS
GCS_PROJECT_ID  = os.getenv("GCS_PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_RAW_TRAIN = f"gs://{GCS_BUCKET_NAME}/data/raw_data/fintell_train.parquet"
GCS_RAW_VAL   = f"gs://{GCS_BUCKET_NAME}/data/raw_data/fintell_val.parquet"

# GCS PREFIXES
GCS_PREFIX_DL_DATA      = "dl_data"
GCS_PREFIX_DL_MODELS    = "dl_models"
GCS_PREFIX_DL_PLOTS     = "dl_plots"
GCS_PREFIX_DL_RUNS      = "dl_runs"

GCS_PREFIX_TOPIC_DL_DATA   = "topic_dl_data"
GCS_PREFIX_TOPIC_DL_MODELS = "topic_dl_models"
GCS_PREFIX_TOPIC_DL_PLOTS  = "topic_dl_plots"
GCS_PREFIX_TOPIC_DL_RUNS   = "topic_dl_runs"

# FLAGS MODELS
# MODEL_NAME: svc | logistic_regression | random_forest | naive_bayes
MODEL_NAME = os.getenv("MODEL_NAME", "svc")
# MODEL_DL_NAME: lstm | gru | bigru | bilstm
MODEL_DL_NAME       = os.getenv("MODEL_DL_NAME",       "lstm")
# MODEL_TOPIC_DL_NAME: lstm | gru | bigru | bilstm
MODEL_TOPIC_DL_NAME = os.getenv("MODEL_TOPIC_DL_NAME", "lstm")
# EMBEDDER_NAME: word2vec | keras_embedding
EMBEDDER_NAME       = os.getenv("EMBEDDER_NAME",       "word2vec")

# DL HYPERPARAMS
MAXLEN      = int(os.getenv("MAXLEN", "80"))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "60"))
# USE_CLASS_WEIGHT: true = balanced class weights | false
USE_CLASS_WEIGHT = os.getenv("USE_CLASS_WEIGHT", "false").lower() == "true"

# DEV
# SAMPLE_SIZE: 0 = full dataset | n = nombre de lignes
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "0"))

# PRODUCTION — chemins GCS fixes vers les artefacts déployés
GCS_PROD_SENTIMENT_MODEL     = os.getenv("GCS_PROD_SENTIMENT_MODEL",     "production/sentiments/bigru.keras")
GCS_PROD_SENTIMENT_ENCODER   = os.getenv("GCS_PROD_SENTIMENT_ENCODER",   "production/sentiments/label_encoder.pkl")
GCS_PROD_SENTIMENT_TOKENIZER = os.getenv("GCS_PROD_SENTIMENT_TOKENIZER", "production/sentiments/tokenizer.pkl")

GCS_PROD_TOPIC_MODEL   = os.getenv("GCS_PROD_TOPIC_MODEL",   "production/topic/model_dl_bigru_20260608_145844.keras")
GCS_PROD_TOPIC_ENCODER = os.getenv("GCS_PROD_TOPIC_ENCODER", "production/topic/encoder_20260608_145844.pkl")
GCS_PROD_TOPIC_VOCAB   = os.getenv("GCS_PROD_TOPIC_VOCAB",   "production/topic/vocab_20260608_145844.pkl")
