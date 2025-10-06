import os
import mlflow
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score
from src.utils.io import load_yaml

def main():
    dcfg = load_yaml("configs/data.yaml")
    tcfg = load_yaml("configs/train.yaml")

    mlflow.set_tracking_uri(tcfg["mlflow"]["tracking_uri"])

    test = pd.read_csv(os.path.join(dcfg["paths"]["interim_dir"], "test.csv"))
    y = test[dcfg["target_column"]]
    X = test.drop(columns=[dcfg["target_column"]])

    pre = load(os.path.join(dcfg["paths"]["processed_dir"], "preproc.joblib"))
    Xp = pre.transform(X)

    model = load("models/model.joblib")
    yhat = model.predict(Xp)
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat, average="binary") if len(set(y))==2 else f1_score(y, yhat, average="macro")

    with mlflow.start_run(run_name="eval", nested=True):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

    print({"accuracy": acc, "f1": f1})

if __name__ == "__main__":
    main()
