import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/creditcard.csv")
print("Data loaded. Shape:", df.shape)

# Sample a subset for faster processing
df = df.sample(n=10000, random_state=42)
print("Sampled Data Shape:", df.shape)

# Preprocess data
X = df.drop("Class", axis=1).values
y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

# Train Random Forest model
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=5, max_depth=5, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)
print("Training completed.")