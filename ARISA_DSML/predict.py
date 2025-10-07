from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier
import joblib

from ARISA_DSML.config import MODELS_DIR, PROCESSED_DATA_DIR, target

def load_model_and_params():
    model_path = MODELS_DIR / "catboost_bank.cbm"
    params_path = MODELS_DIR / "model_params.pkl"
    if not model_path.exists() or not params_path.exists():
        raise FileNotFoundError("Trained model or params not found. Run train.py first.")
    model = CatBoostClassifier()
    model.load_model(model_path)
    meta = joblib.load(params_path)
    return model, meta["feature_columns"]

def main():
    df_test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    model, feature_columns = load_model_and_params()

    # Align columns (fill any missing)
    for c in feature_columns:
        if c not in df_test.columns:
            df_test[c] = None
    df_test = df_test[feature_columns]

    preds = model.predict(df_test)
    probs = model.predict_proba(df_test)

    out = df_test.copy()
    out[target] = preds
    out["predicted_probability"] = [p[1] for p in probs]
    out_path = MODELS_DIR / "preds.csv"
    out.to_csv(out_path, index=False)
    print(f\"Predictions saved to: {out_path}\")

if __name__ == "__main__":
    main()