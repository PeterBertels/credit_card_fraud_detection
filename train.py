"""
train.py
--------
End-to-end training pipeline for the Credit Card Fraud Detection project.

Steps:
  1. Load and sample the dataset
  2. Visualize class imbalance
  3. Split, scale, and apply SMOTE
  4. Train Random Forest (sklearn) — primary model
  5. Evaluate and save visualizations
  6. Save trained model and scaler to model/

Usage:
    python train.py

Requirements:
    pip install -r requirements.txt
    Dataset: data/creditcard.csv  (download from Kaggle)
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.preprocess import load_data, sample_data, split_and_scale, smote_oversample
from src.visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_feature_distributions,
)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH    = "data/creditcard.csv"
MODEL_DIR    = "model"
SAMPLE_SIZE  = 50_000
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  Credit Card Fraud Detection — Training Pipeline")
    print("=" * 60)

    # 1. Load & sample
    df = load_data(DATA_PATH)
    df = sample_data(df, SAMPLE_SIZE)

    # 2. Visualize raw class imbalance
    plot_class_distribution(df["Class"].values)
    plot_feature_distributions(df, features=["V1", "V2", "V3", "V14"])

    # 3. Split, scale, oversample
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    X_train_bal, y_train_bal = smote_oversample(X_train, y_train)

    # 4. Train Random Forest
    print("\nTraining Random Forest (sklearn)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_bal, y_train_bal)
    print("Training complete.")

    # 5. Evaluate
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print("\n" + "─" * 40)
    print("Classification Report — Random Forest")
    print("─" * 40)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # 6. Visualizations
    plot_confusion_matrix(y_test, y_pred, model_name="Random Forest")
    plot_roc_curve(y_proba=y_proba, y_true=y_test, model_name="Random Forest")
    plot_feature_importance(feature_names, rf.feature_importances_,
                            model_name="Random Forest")

    # 7. Save model artifacts
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print("\nModel artifacts saved to model/")
    print("Visualizations saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
