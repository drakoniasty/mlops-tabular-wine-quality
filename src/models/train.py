import os
import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.utils.io import load_yaml

def main():
    dcfg = load_yaml("configs/data.yaml")
    tcfg = load_yaml("configs/train.yaml")

    mlflow.set_tracking_uri(tcfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(tcfg["experiment_name"])

    train = pd.read_csv(os.path.join(dcfg["paths"]["interim_dir"], "train.csv"))
    target = dcfg["target_column"]
    X_train = train.drop(columns=[target])
    y_train = train[target]

    preproc = load(os.path.join(dcfg["paths"]["processed_dir"], "preproc.joblib"))
    Xtr = preproc.transform(X_train)

    # model z configa (tu na sztywno RF dla zwięzłości)
    params = tcfg["model"]["params"]
    model = RandomForestClassifier(**params)

    with mlflow.start_run(run_name=tcfg["run_name"]):
        mlflow.log_params(params)
        model.fit(Xtr, y_train)

        # Zapisz model + preproc jako artefakty
        out_dir = "models"
        os.makedirs(out_dir, exist_ok=True)
        dump(model, os.path.join(out_dir, "model.joblib"))
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(os.path.join(dcfg["paths"]["processed_dir"], "preproc.joblib"))

        print("Training done.")

if __name__ == "__main__":
    main()
