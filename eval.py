# Metrics, PR/ROC, confusion matrices
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score, precision_recall_fscore_support
)
from matplotlib import pyplot as plt

def binarize_scores(scores, threshold):
    return (scores >= threshold).astype(int)

def metrics_from_scores(y_true, scores):
    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    p, r, _ = precision_recall_curve(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "precision_curve": p, "recall_curve": r,
            "fpr": fpr, "tpr": tpr}

def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return cm, pr, rc, f1

# def ensemble_score(xgb_p, rf_p, iso_s, ae_s, w=(0.35,0.25,0.20,0.20)):
#     w1,w2,w3,w4 = w
#     # normalize anomaly scores to [0,1] via rank (robust for heavy-tailed)
#     import scipy.stats as st
#     iso_n = st.rankdata(iso_s)/len(iso_s)
#     ae_n  = st.rankdata(ae_s)/len(ae_s)
#     return w1*xgb_p + w2*rf_p + w3*iso_n + w4*ae_n

def plot_confusion_matrix(
    cm,
    title="Confusion Matrix",
    classes=("Legit (0)", "Fraud (1)"),
    figsize=(3.2, 2.6),     # smaller default
    cmap="Blues",
    normalize=False,        # no percentage math unless you turn it on
    show_percent=False      # <-- default OFF
):
    cm_pct = None
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct = cm.astype(np.float64) / row_sums

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:d}"
            if show_percent and cm_pct is not None:
                text += f"\n({cm_pct[i, j]*100:.1f}%)"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10, fontweight="bold"
            )

    fig.tight_layout()
    return fig
