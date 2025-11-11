# #Uses SHAP + LLM prompt 

# explainer.py
from __future__ import annotations
import json
import textwrap
from typing import Callable, Dict, List, Tuple

import numpy as np

# ---- Optional SHAP import (kept inside functions if you prefer) --------------
import shap

# =========================
# SHAP: Top feature picker
# =========================

def _sanitize_xgb_base_score(xgb_model) -> None:
    """Fixes xgboost booster attr base_score like '[5E-1]' -> '5E-1' so SHAP can parse it."""
    try:
        booster = xgb_model.get_booster()
        bs = booster.attr("base_score")
        if bs:
            s = bs.strip()
            if s.startswith("[") and s.endswith("]"):
                booster.set_attr(base_score=s[1:-1])
    except Exception:
        # non-fatal; SHAP fallback path can still work
        pass

def shap_top_features(model, X_sample_df, feature_names: List[str], max_k: int = 5) -> List[Tuple[str, float, float]]:
    """
    Return [(feature_name, shap_value, abs_shap)] for a SINGLE row in X_sample_df.
    Tries fast TreeExplainer; falls back to model-agnostic Explainer if needed.
    """
    # Try the fast TreeExplainer first
    try:
        _sanitize_xgb_base_score(model)
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_sample_df)
        # RF may return list [class0, class1]; take positive class
        if isinstance(vals, list):
            vals = vals[1]
        shap_row = np.asarray(vals)[0]  # (n_features,)
    except Exception:
        # Robust fallback: model-agnostic SHAP using predict_proba
        masker = shap.maskers.Independent(X_sample_df, max_samples=100)
        f = lambda X: model.predict_proba(X)[:, 1]
        explainer = shap.Explainer(f, masker)
        sv = explainer(X_sample_df, max_evals=200)
        shap_row = np.asarray(sv.values[0])

    order = np.argsort(np.abs(shap_row))[::-1][:max_k]
    return [(feature_names[i], float(shap_row[i]), float(abs(shap_row[i]))) for i in order]


# =========================
# Deterministic explainer
# =========================

def fallback_explanation(
    transaction_id: int | str,
    proba_dict: Dict[str, float],
    anomaly_dict: Dict[str, float],
    shap_dict: List[Tuple[str, float, float]] | None,
) -> str:
    """
    Clear, short, no-LLM explanation text.
    """
    lines = [f"Transaction #{transaction_id}: self-explaining report"]
    if "xgb" in proba_dict:
        lines.append(f"- XGBoost fraud probability: {proba_dict['xgb']:.4f}")
    if "rf" in proba_dict:
        lines.append(f"- RandomForest fraud probability: {proba_dict['rf']:.4f}")
    if "iso" in anomaly_dict:
        lines.append(f"- IsolationForest anomaly score: {anomaly_dict['iso']:.4f}")
    if "ae" in anomaly_dict:
        lines.append(f"- Autoencoder reconstruction MSE: {anomaly_dict['ae']:.6f}")

    if shap_dict:
        lines.append("- Top contributing features (SHAP on XGBoost):")
        for name, val, mag in shap_dict[:5]:
            direction = "↑ increases fraud risk" if val > 0 else "↓ decreases"
            lines.append(f"   • {name}: SHAP={val:+.4f} ({direction})")

    lines.append("")
    lines.append("Triage:")
    lines.append("- Step-up verification for this user (OTP / call-back).")
    lines.append("- Rate-limit repeated amounts and short-interval bursts for 24h.")
    lines.append("- Review merchant/device linkage for multi-user fanout.")
    return "\n".join(lines)


# =========================
# Ollama client + helpers
# =========================

def is_ollama_up(host: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    import requests
    url = host.rstrip("/") + "/api/tags"
    try:
        requests.get(url, timeout=timeout)
        return True
    except Exception:
        return False

def make_ollama_client(model: str = "phi3:mini", host: str = "http://localhost:11434") -> Callable[[str], str]:
    """
    Returns a callable llm(prompt:str)->str using Ollama's /api/generate (non-streaming).
    """
    import requests
    url = host.rstrip("/") + "/api/generate"

    def _call(prompt: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 300, "temperature": 0.2},
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        txt = r.json().get("response", "") or ""
        # light cleanup
        return textwrap.shorten(txt.replace("\r", " ").strip(), width=1200, placeholder="…")

    return _call


# =========================
# LLM explainer wrapper
# =========================

_PROMPT = (
    "You are a concise fraud analyst. Using the signals below, explain in <=80 words "
    "why this transaction was flagged. Then list 3 single-line triage actions as bullets. "
    "Return plain text only (no JSON).\n\n"
    "Signals:\n"
    "- XGBoost prob: {xgb}\n"
    "- RandomForest prob: {rf}\n"
    "- IsolationForest anomaly: {iso}\n"
    "- Autoencoder MSE: {ae}\n"
    "- Top features (name: SHAP): {top_feats}\n"
)

def llm_explain_or_fallback(
    transaction_id: int | str,
    proba_dict: Dict[str, float],
    anomaly_dict: Dict[str, float],
    shap_dict: List[Tuple[str, float, float]] | None,
    feature_row_df,  # kept for parity; not used by the LLM prompt directly
    llm_client: Callable[[str], str] | None = None,
) -> str:
    """
    If llm_client is provided (e.g., make_ollama_client(...)) and reachable, use it.
    Otherwise return deterministic fallback_explanation(...).
    """
    if llm_client is None:
        return fallback_explanation(transaction_id, proba_dict, anomaly_dict, shap_dict)

    # Prepare short top-features string: "V14:+0.87, V12:+0.45, ..."
    if shap_dict:
        top_feats = ", ".join(f"{n}:{v:+.3f}" for n, v, _ in shap_dict[:5])
    else:
        top_feats = "n/a"

    prompt = _PROMPT.format(
        xgb=f"{proba_dict.get('xgb', float('nan')):.4f}",
        rf=f"{proba_dict.get('rf', float('nan')):.4f}",
        iso=f"{anomaly_dict.get('iso', float('nan')):.4f}",
        ae=f"{anomaly_dict.get('ae', float('nan')):.6f}",
        top_feats=top_feats,
    )

    try:
        txt = llm_client(prompt).strip()
        return txt if txt else fallback_explanation(transaction_id, proba_dict, anomaly_dict, shap_dict)
    except Exception as e:
        fb = fallback_explanation(transaction_id, proba_dict, anomaly_dict, shap_dict)
        return f"LLM error: {e}\n\n{fb}"
