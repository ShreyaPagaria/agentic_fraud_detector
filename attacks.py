# attacks.py
"""
Stress tests limited to:
  1) Noise injection on features
  2) Fraud camouflage (make a fraction of fraud rows look legit)

Used by app.py:
  - run_attacks_and_eval(models, X_train, X_test, y_train, y_test, feature_names)
  - flatten_results(results)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Any
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from eval import metrics_from_scores  # reuse your helper for PR/ROC curves etc.

# -----------------------
# Attack generators
# -----------------------
def noise_attack(X: pd.DataFrame, scale: float = 0.05) -> pd.DataFrame:
    """
    Add Gaussian noise to numeric features (std = `scale`).
    Keeps non-numeric columns unchanged.
    """
    Xn = X.copy()
    numcols = Xn.select_dtypes(include=[np.number]).columns
    noise = np.random.normal(0, scale, size=(len(Xn), len(numcols)))
    Xn[numcols] = Xn[numcols].values + noise
    return Xn

def camouflage_attack(X: pd.DataFrame, y: pd.Series, frac: float = 0.3) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Make a fraction of FRAUD rows look like LEGIT by replacing their feature
    vectors with the median of legitimate rows. Labels stay the same (1),
    so detectors face 'camouflaged' fraud.
    """
    Xc = X.reset_index(drop=True).copy()
    yc = pd.Series(y).reset_index(drop=True).copy()

    fraud_idx = np.where(yc == 1)[0]
    if len(fraud_idx) == 0:
        return Xc, yc

    n = max(1, int(len(fraud_idx) * frac))
    to_cam = np.random.choice(fraud_idx, n, replace=False)

    legit_meds = Xc[yc == 0].median()
    Xc.loc[to_cam, :] = legit_meds.values
    return Xc, yc

# -----------------------
# Evaluation helper
# -----------------------
# def evaluate_models_on_set(models: Tuple[Any, Any, Any, Any],
#                            X: pd.DataFrame,
#                            y: pd.Series,
#                            feature_names):
#     """
#     Evaluate xgb, rf, iso, ae on provided X, y.
#     Returns dict[name] = metrics dict (pr_auc, roc_auc, cm, precision, recall, f1, threshold).
#     """
#     from models import predict_all  # local import avoids circulars
#     xgb_p, rf_p, iso_s, ae_s = predict_all(models[0], models[1], models[2], models[3], X, feature_names)

#     out = {}
#     # Supervised (fixed 0.5 threshold for demo)
#     for name, scores in [("xgb", xgb_p), ("rf", rf_p)]:
#         m = metrics_from_scores(y, scores)
#         thr = 0.5
#         yb = (scores >= thr).astype(int)
#         cm = confusion_matrix(y, yb)
#         pr, rc, f1, _ = precision_recall_fscore_support(y, yb, average="binary", zero_division=0)
#         m.update({"cm": cm, "precision": float(pr), "recall": float(rc), "f1": float(f1), "threshold": float(thr)})
#         out[name] = m

#     # Unsupervised: treat higher score as more anomalous; use 99th percentile as default threshold
#     for name, scores in [("iso", iso_s), ("ae", ae_s)]:
#         m = metrics_from_scores(y, scores)
#         thr = float(np.percentile(scores, 99))
#         yb = (scores >= thr).astype(int)
#         cm = confusion_matrix(y, yb)
#         pr, rc, f1, _ = precision_recall_fscore_support(y, yb, average="binary", zero_division=0)
#         m.update({"cm": cm, "precision": float(pr), "recall": float(rc), "f1": float(f1), "threshold": float(thr)})
#         out[name] = m

#     return out
def evaluate_models_on_set(models: Tuple[Any, Any, Any, Any],
                           X: pd.DataFrame,
                           y: pd.Series,
                           feature_names):
    """
    Evaluate xgb, rf, iso, ae on provided X, y.
    Returns dict[name] = metrics dict (pr_auc, roc_auc, cm, precision, recall, f1, threshold).
    """
    from models import predict_all  # local import avoids circulars

    # Your predict_all signature is predict_all(xgb, rf, iso, ae, X)
    xgb_p, rf_p, iso_s, ae_s = predict_all(models[0], models[1], models[2], models[3], X)

    out = {}
    # Supervised (fixed 0.5 threshold for demo)
    for name, scores in [("xgb", xgb_p), ("rf", rf_p)]:
        m = metrics_from_scores(y, scores)
        thr = 0.5
        yb = (scores >= thr).astype(int)
        cm = confusion_matrix(y, yb)
        pr, rc, f1, _ = precision_recall_fscore_support(y, yb, average="binary", zero_division=0)
        m.update({"cm": cm, "precision": float(pr), "recall": float(rc), "f1": float(f1), "threshold": float(thr)})
        out[name] = m

    # Unsupervised: use 99th percentile as default threshold
    for name, scores in [("iso", iso_s), ("ae", ae_s)]:
        m = metrics_from_scores(y, scores)
        thr = float(np.percentile(scores, 99))
        yb = (scores >= thr).astype(int)
        cm = confusion_matrix(y, yb)
        pr, rc, f1, _ = precision_recall_fscore_support(y, yb, average="binary", zero_division=0)
        m.update({"cm": cm, "precision": float(pr), "recall": float(rc), "f1": float(f1), "threshold": float(thr)})
        out[name] = m

    return out


# -----------------------
# Run selected attacks and collect results
# -----------------------
def run_attacks_and_eval(models,
                         X_train, X_test,
                         y_train, y_test,
                         feature_names,
                         random_seed: int = 42):
    """
    Scenarios returned:
      - 'clean'
      - 'noise_5pct'  (Gaussian noise std=0.05)
      - 'camouflage'  (30% of fraud rows camouflaged)
    """
    np.random.seed(random_seed)
    results = {}

    # Baseline (clean test)
    results["clean"] = evaluate_models_on_set(models, X_test, y_test, feature_names)

    # Noise attack (5%)
    X_noisy = noise_attack(X_test, scale=0.05)
    results["noise_5pct"] = evaluate_models_on_set(models, X_noisy, y_test, feature_names)

    # Fraud camouflage (30% of fraud rows)
    X_cam, y_cam = camouflage_attack(X_test, y_test, frac=0.30)
    results["camouflage"] = evaluate_models_on_set(models, X_cam, y_cam, feature_names)

    return results

# -----------------------
# Tidy results for display
# -----------------------
def flatten_results(results: dict) -> pd.DataFrame:
    rows = []
    for scenario, per_model in results.items():
        if not isinstance(per_model, dict):
            continue
        for model_name, m in per_model.items():
            rows.append({
                "scenario": scenario,
                "model": model_name,
                "pr_auc": float(m.get("pr_auc", np.nan)),
                "roc_auc": float(m.get("roc_auc", np.nan)),
                "precision": float(m.get("precision", np.nan)),
                "recall": float(m.get("recall", np.nan)),
                "f1": float(m.get("f1", np.nan)),
                "threshold": float(m.get("threshold", np.nan)) if m.get("threshold") is not None else np.nan,
            })
    return pd.DataFrame(rows)
