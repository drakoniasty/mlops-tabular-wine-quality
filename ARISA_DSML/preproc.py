import os
from pathlib import Path
from loguru import logger
import pandas as pd

from ARISA_DSML.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, target, categorical

def _read_csv_any(path: Path) -> pd.DataFrame:
    # Try comma first (as seen in your files), then semicolon fallback.
    try:
        df = pd.read_csv(path, sep=",")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path, sep=";")
    return df

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip spaces from column names
    df.columns = [c.strip() for c in df.columns]
    return df

def _ensure_target_binary(df: pd.DataFrame) -> pd.DataFrame:
    if target in df.columns:
        # Map typical yes/no to 1/0, keep numeric if already numeric
        if df[target].dtype == "object":
            df[target] = df[target].str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(df[target])
        # If still not numeric, try to coerce
        if not pd.api.types.is_numeric_dtype(df[target]):
            df[target] = pd.to_numeric(df[target], errors="ignore")
    return df

def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    df = _ensure_target_binary(df)

    # Convert obvious numeric columns if present
    for col in ["age","balance","day","duration","campaign","pdays","previous"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Minimal NA handling
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].astype(str).str.strip().replace({"nan": None}).fillna("unknown")

    return df

def preprocess_all()->None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "train.csv": RAW_DATA_DIR / "bank-full.csv",
        "val.csv": RAW_DATA_DIR / "bank-full copy.csv",
        "test.csv": RAW_DATA_DIR / "bank-full copy 2.csv",
    }

    for out_name, in_path in files.items():
        if not in_path.exists():
            logger.error(f"Missing input file: {in_path}")
            continue
        df = _read_csv_any(in_path)
        df = _basic_clean(df)

        # Ensure consistent column order across splits
        df = df.reindex(sorted(df.columns), axis=1)
        df.to_csv(PROCESSED_DATA_DIR / out_name, index=False)
        logger.info(f"Saved processed -> {PROCESSED_DATA_DIR / out_name} (shape={df.shape})")

if __name__ == "__main__":
    logger.info(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
    preprocess_all()