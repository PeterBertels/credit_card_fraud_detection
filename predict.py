"""
predict.py
----------
Load the trained model and scaler and run predictions on new transactions.

Can be used as:
  - A standalone CLI script:   python predict.py
  - An importable module:      from predict import predict

The predict() function is also called by the HTML frontend (app.html)
via a local server when deployed.

Usage (CLI):
    python predict.py
"""

import os
import pickle
import numpy as np

MODEL_PATH   = "model/model.pkl"
SCALER_PATH  = "model/scaler.pkl"
FEATURES_PATH = "model/feature_names.pkl"


def load_artifacts():
    """Load and return the trained model, scaler, and feature names."""
    for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artifact: {path}\n"
                "Run train.py first to generate model artifacts."
            )
    with open(MODEL_PATH,    "rb") as f: model = pickle.load(f)
    with open(SCALER_PATH,   "rb") as f: scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: feature_names = pickle.load(f)
    return model, scaler, feature_names


def predict(features: list) -> dict:
    """
    Run a fraud prediction on a single transaction.

    Parameters
    ----------
    features : list of float
        30 numeric values in order: [Time, V1, V2, ..., V28, Amount]

    Returns
    -------
    dict with keys:
        is_fraud   : bool
        label      : "Fraud" | "Legitimate"
        confidence : float (0–100, model confidence in its prediction)
    """
    if len(features) != 30:
        raise ValueError(f"Expected 30 features, received {len(features)}.")

    model, scaler, _ = load_artifacts()
    X = np.array(features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])
    proba      = float(model.predict_proba(X_scaled)[0][prediction])

    return {
        "is_fraud":   prediction == 1,
        "label":      "Fraud" if prediction == 1 else "Legitimate",
        "confidence": round(proba * 100, 2),
    }


if __name__ == "__main__":
    # Example: a near-zero legitimate transaction
    example_legit = [0.0] + [0.0] * 28 + [50.0]
    result = predict(example_legit)
    print(f"Prediction  : {result['label']}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"Is Fraud    : {result['is_fraud']}")
