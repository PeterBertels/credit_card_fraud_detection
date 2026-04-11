"""
src/visualize.py
----------------
Visualization helpers: confusion matrix, ROC curve,
feature importance chart, and class distribution plot.
All figures are saved to the outputs/ directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_class_distribution(y, title="Class Distribution", filename="class_distribution.png"):
    """Bar chart showing fraud vs. legitimate counts."""
    labels, counts = np.unique(y, return_counts=True)
    names = ["Legitimate" if l == 0 else "Fraud" for l in labels]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, counts, color=["#4a90d9", "#e74c3c"], width=0.4)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{count:,}", ha="center", va="bottom", fontsize=10)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_confusion_matrix(y_true, y_pred, model_name="Model", filename="confusion_matrix.png"):
    """Heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_roc_curve(y_true, y_proba, model_name="Model", filename="roc_curve.png"):
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#e74c3c", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.fill_between(fpr, tpr, alpha=0.08, color="#e74c3c")
    plt.title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}  (AUC = {roc_auc:.4f})")
    return roc_auc


def plot_feature_importance(feature_names, importances, model_name="Random Forest",
                             filename="feature_importance.png", top_n=15):
    """Horizontal bar chart of top-N feature importances."""
    indices = np.argsort(importances)[-top_n:]
    names   = [feature_names[i] for i in indices]
    values  = importances[indices]

    plt.figure(figsize=(8, 6))
    bars = plt.barh(names, values, color="#4a90d9")
    plt.title(f"Feature Importance — {model_name} (top {top_n})",
              fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_feature_distributions(df, features=("V1", "V2", "V3"),
                                filename="feature_distributions.png"):
    """Histogram + KDE for a set of features."""
    n = len(features)
    plt.figure(figsize=(10, 3 * n))
    for i, feat in enumerate(features, 1):
        plt.subplot(n, 1, i)
        sns.histplot(df[feat], bins=50, kde=True, color="#4a90d9")
        plt.title(f"Distribution of {feat}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")
