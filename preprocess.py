# Load, clean, scale and split the dataset
# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC_TO_SCALE = ["Amount"]  # we drop 'Time' if present

def _read_any(source):
    """source can be a file-like (UploadedFile) or a path string."""
    if hasattr(source, "read"):
        source.seek(0)
        return pd.read_csv(source)
    return pd.read_csv(source)

def load_and_preprocess(source):
    df = _read_any(source)

    # Basic checks
    if df is None or df.empty:
        raise ValueError("Uploaded CSV is empty.")
    if "Class" not in df.columns:
        raise ValueError("CSV must contain a 'Class' column with 0/1 labels.")

    # Ensure labels are clean ints
    df = df.dropna(subset=["Class"]).copy()
    # Some Kaggle zips load Class as float; coerce to int safely
    df["Class"] = df["Class"].astype(int)

    # Drop Time if present
    X = df.drop(columns=["Class", "Time"], errors="ignore")
    y = df["Class"].values

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Fit scalers on TRAIN only, then transform train & test
    scalers = {}
    for col in NUMERIC_TO_SCALE:
        if col in X_train.columns:
            sc = StandardScaler()
            X_train[col] = sc.fit_transform(X_train[col].values.reshape(-1, 1))
            X_test[col]  = sc.transform(X_test[col].values.reshape(-1, 1))
            scalers[col] = sc

    feature_names = list(X.columns)
    return df, X_train, X_test, y_train, y_test, scalers, feature_names
