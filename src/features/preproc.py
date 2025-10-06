import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
from src.utils.io import load_yaml, ensure_dir
import os

def infer_columns(df, target):
    num = df.drop(columns=[target]).select_dtypes(include=["int64","float64"]).columns.tolist()
    cat = df.drop(columns=[target]).select_dtypes(exclude=["int64","float64"]).columns.tolist()
    return num, cat

def main():
    dcfg = load_yaml("configs/data.yaml")
    pcfg = load_yaml("configs/train.yaml")

    interim = dcfg["paths"]["interim_dir"]
    processed = dcfg["paths"]["processed_dir"]
    ensure_dir(processed)

    train = pd.read_csv(os.path.join(interim, "train.csv"))
    target = dcfg["target_column"]
    num, cat = infer_columns(train, target)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ],
        remainder="drop"
    )

    pipe = Pipeline(steps=[("pre", pre)])
    X = train.drop(columns=[target])
    pipe.fit(X)  # tylko fit na train

    dump(pipe, os.path.join(processed, "preproc.joblib"))
    print("Saved preprocessor with:", len(num), "num and", len(cat), "cat features.")

if __name__ == "__main__":
    main()
