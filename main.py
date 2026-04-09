import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/creditcard.csv")
print("Data loaded. Shape:", df.shape)

# Sample data
df = df.sample(n=50000, random_state=42)
print("Sampled Data Shape:", df.shape)

# Basic info
print("\nDataset Info:")
print(df.info())
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Visualize class imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Class Distribution (0: Non-Fraud, 1: Fraud)")
plt.savefig("class_distribution.png")
plt.close()

# Visualize feature distributions (V1-V3)
plt.figure(figsize=(10, 6))
for i in range(1, 4):
    plt.subplot(3, 1, i)
    sns.histplot(df[f"V{i}"], bins=50, kde=True)
    plt.title(f"Distribution of V{i}")
plt.tight_layout()
plt.savefig("feature_distributions.png")
plt.close()

# Preprocess data
X = df.drop("Class", axis=1).values
y = df["Class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE - X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

# Train model
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)
print("Training completed.")

# Evaluate model
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

# Feature importance
feature_importance = pd.DataFrame({'feature': df.drop('Class', axis=1).columns, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

print("\nVisualizations saved: class_distribution.png, feature_distributions.png, confusion_matrix.png, roc_curve.png, feature_importance.png")