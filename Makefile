.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

run_preprocess:
	python -c "import pandas as pd; from fintell_package.main import preprocess_sentiment; from fintell_package.params import RAW_DIR; preprocess_sentiment(pd.read_parquet(RAW_DIR / 'fintell_train.parquet'), split='train'); preprocess_sentiment(pd.read_parquet(RAW_DIR / 'fintell_val.parquet'), split='val')"

run_train:
	python -c "from fintell_package.main import train; train()"

run_evaluate:
	python -c "from fintell_package.main import evaluate; evaluate()"

run_all:
	python fintell_package/main.py
