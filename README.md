# Agentic Financial Fraud Detector

An interactive **Streamlit web app** that detects and explains financial fraud using a combination of **machine learning**, **anomaly detection**, and **LLM-based interpretability**.

---

## Overview

This project demonstrates an **agentic AI pipeline** for fraud detection that integrates:
- **Supervised models:** `XGBoost`, `Random Forest`
- **Unsupervised models:** `Isolation Forest`, `Autoencoder`
- **Explainability:** SHAP feature attribution + optional natural-language explanations via **Ollama LLM** [in progress]
- **UI:** Built with Streamlit for interactive exploration and analysis
- **Adversarial Attacks:** Evaluate robustness with two lightweight attacks (Noise Injection and Fraud Camouflage)

---

## Key Features

✅ Detects fraudulent transactions from the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
✅ Visualizes confusion matrices and key metrics (Precision, Recall, ROC-AUC, PR-AUC)  
✅ Explains model reasoning using **SHAP** values  
✅ Provides natural-language transaction analysis powered by **Ollama** (e.g. `phi3:mini`)  

---

## Project Structure - for easier understanding of the directory
fraud_agent/
├── app.py               # Streamlit UI entry point
├── agent.py             # Pipeline integration logic
├── attacks.py           # Noise & Camouflage attacks + evaluation helpers 
├── eval.py              # Metrics & confusion matrix plotting
├── explainer.py         # SHAP + LLM/Ollama explainer
├── models.py            # Model training/loading utilities
├── preprocess.py        # Data loading & preprocessing
├── data/                # (optional) dataset folder
├── models/              # (optional) saved models
└── requirements.txt     # Python dependencies

