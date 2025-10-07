# ===== Makefile for Bank Marketing MLflow Project =====

PYTHON := python
PIP := $(PYTHON) -m pip

# ========= BASIC SETUP =========

.PHONY: requirements preprocess train predict resolve lint clean

# Install dependencies
requirements:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run linting checks
lint:
	flake8 .
	black --check .

# ========= PIPELINE =========

# Step 1: preprocess all CSVs in data/raw -> data/processed
preprocess:
	$(PYTHON) ARISA_DSML/preproc.py

# Step 2: train model and log to MLflow
train:
	MLFLOW_TRACKING_URI=$$MLFLOW_TRACKING_URI \
	$(PYTHON) ARISA_DSML/train.py

# Step 3: resolve MLflow champion/challenger before predictions
resolve:
	MLFLOW_TRACKING_URI=$$MLFLOW_TRACKING_URI \
	$(PYTHON) ARISA_DSML/resolve.py

# Step 4: run predictions on processed test.csv
predict:
	MLFLOW_TRACKING_URI=$$MLFLOW_TRACKING_URI \
	$(PYTHON) ARISA_DSML/predict.py

# ========= UTILITIES =========

# Remove artifacts and temporary files
clean:
	rm -rf __pycache__ .pytest_cache models/*.pkl models/*.cbm models/preds.csv reports/figures/*.png
