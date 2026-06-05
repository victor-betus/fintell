.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

run_preprocess_train:
	python -c "from fintell_package.main import preprocess_sentiment; preprocess_sentiment(split='train')"

run_preprocess_val:
	python -c "from fintell_package.main import preprocess_sentiment; preprocess_sentiment(split='val')"

run_train:
	python -c "from fintell_package.main import train; train()"

run_evaluate:
	python -c "from fintell_package.main import evaluate; evaluate()"

run_all:
	$(MAKE) run_preprocess_train
	$(MAKE) run_preprocess_val
	$(MAKE) run_train
	$(MAKE) run_evaluate
