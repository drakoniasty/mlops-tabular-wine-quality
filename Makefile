.PHONY: setup data split featurize train eval all lint test

setup:
	pip install -r requirements.txt

lint:
	ruff check . && black --check .

test:
	pytest -q

data:
	python -m src.data.download

split:
	python -m src.data.split

featurize:
	python -m src.features.preproc

train:
	python -m src.models.train

eval:
	python -m src.models.evaluate

all: data split featurize train eval
