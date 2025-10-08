from pathlib import Path
from catboost import CatBoostClassifier, Pool, cv
import joblib
from loguru import logger
import mlflow
import optuna
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split

from ARISA_DSML.config import FIGURES_DIR, MODEL_NAME, MODELS_DIR, PROCESSED_DATA_DIR, target, categorical
from ARISA_DSML.helpers import get_git_commit_hash

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def detect_categorical(df: pd.DataFrame) -> list[str]:
    # Auto-detect object dtype as categoricals plus known list (dedup preserve order)
    auto = [c for c in df.columns if df[c].dtype == "object"]
    pref = [c for c in categorical if c in df.columns]
    seen = set()
    out = []
    for c in pref + auto:
        if c not in seen and c in df.columns:
            out.append(c)
            seen.add(c)
    return out

def run_hyperopt(X_train, y_train, categorical_indices, n_trials:int=20):
    best_params_path = MODELS_DIR / "best_params.pkl"
    if best_params_path.exists():
        return best_params_path

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 100, 600),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 100.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 50.0, log=True),
            "loss_function": "Logloss",
            "eval_metric": "F1",
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 0
        }
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=categorical_indices,
            early_stopping_rounds=100,
            verbose=0
        )
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)
        f1 = f1_score(y_val, preds)
        logloss = log_loss(y_val, probs)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("logloss", logloss)
        return model.get_best_score()["validation"]["Logloss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study.best_params, best_params_path)
    logger.info(f"Best params: {study.best_params}")
    return best_params_path

def train_cv(X, y, categorical_indices, params, n_folds:int=5):
    params = {**params}
    params.update({
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "auto_class_weights": "Balanced",
        "random_seed": 42
    })
    data = Pool(X, y, cat_features=categorical_indices)
    cv_results = cv(
        params=params,
        pool=data,
        fold_count=n_folds,
        partition_random_seed=42,
        shuffle=True,
        plot=False
    )
    out = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(out, index=False)
    return out

def plot_error_scatter(df_plot, x="iterations", y="test-F1-mean", err="test-F1-std",
                       name="", title="", xtitle="", ytitle="", yaxis_range=None):
    fig = go.Figure()
    name = name or y
    fig.add_trace(go.Scatter(x=df_plot[x], y=df_plot[y], mode="lines", name=name, line={"color": "blue"}))
    # Correct polygon (x with reversed x; y with upper then reversed lower)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_plot[x], df_plot[x][::-1]]),
        y=pd.concat([df_plot[y] + df_plot[err], (df_plot[y] - df_plot[err])[::-1]]),
        fill="toself",
        fillcolor="rgba(0, 0, 255, 0.2)",
        line={"color": "rgba(255,255,255,0)"},
        showlegend=False
    ))
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, template="plotly_white")
    if yaxis_range is not None:
        fig.update_layout(yaxis={"range": yaxis_range})
    fig.write_image((FIGURES_DIR / f"{y}_vs_{x}.png"))
    return fig

def train_full(X, y, categorical_indices, params, cv_results=None, artifact_name="catboost_bank.cbm"):
    params = {**(params or {})}
    params.update({
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": 100
    })
    model = CatBoostClassifier(**params)
    model.fit(X, y, cat_features=categorical_indices, verbose=100, early_stopping_rounds=200)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / artifact_name
    model.save_model(model_path)
    import mlflow.catboost
    from pathlib import Path

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(study.best_params, best_params_path)
    mlflow.catboost.log_model(
        cb,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )
    joblib.dump({"feature_columns": list(X.columns), "params": params}, MODELS_DIR / "model_params.pkl")

    # Plot (if cv provided)
    if cv_results is not None and len(cv_results):
        try:
            fig = plot_error_scatter(
                df_plot=cv_results,
                name="Mean F1 Score",
                title="Cross-Validation Mean F1 with Error Bands",
                xtitle="Iterations",
                ytitle="F1",
                yaxis_range=[0.3, 1.0],
            )
        except Exception as e:
            logger.warning(f"Failed to plot CV curves: {e}")

    return model_path

if __name__ == "__main__":
    # Load processed splits
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    df_val = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    # Separate target
    if target not in df_train.columns:
        raise ValueError(f"Target column '{target}' not found in train.csv")
    y_train = df_train.pop(target)
    X_train = df_train

    # Detect categoricals
    cat_cols = detect_categorical(X_train)
    categorical_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    # Hyperopt
    best_params_path = run_hyperopt(X_train, y_train, categorical_indices, n_trials=25)
    params = joblib.load(best_params_path)

    # CV
    cv_path = train_cv(X_train, y_train, categorical_indices, params, n_folds=5)
    cv_results = pd.read_csv(cv_path)

    # Train on (train + val) for final model
    if target in df_val.columns:
        y_val = df_val.pop(target)
        X_val = df_val
        X_all = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    else:
        X_all, y_all = X_train, y_train

    model_path = train_full(X_all, y_all, categorical_indices, params, cv_results=cv_results)
    print(f"Model saved to: {model_path}")