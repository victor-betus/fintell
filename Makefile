.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

run_preprocess_train:
	python -c "from fintell_package.ml_logic.main_sentiment import preprocess_sentiment; preprocess_sentiment(split='train')"

run_preprocess_val:
	python -c "from fintell_package.ml_logic.main_sentiment import preprocess_sentiment; preprocess_sentiment(split='val')"

run_train:
	python -c "from fintell_package.ml_logic.main_sentiment import train; train()"

run_evaluate:
	python -c "from fintell_package.ml_logic.main_sentiment import evaluate; evaluate()"

run_all:
	$(MAKE) run_preprocess_train
	$(MAKE) run_preprocess_val
	$(MAKE) run_train
	$(MAKE) run_evaluate
