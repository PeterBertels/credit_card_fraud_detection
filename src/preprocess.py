"""
src/preprocess.py
-----------------
Data loading, sampling, splitting, scaling, and SMOTE oversampling.
All preprocessing steps are collected here so train.py stays clean.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV from disk."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def sample_data(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Draw a stratified random sample to keep class proportions."""
    sampled = df.sample(n=n, random_state=RANDOM_STATE)
    print(f"Sampled {len(sampled):,} rows. Fraud rate: "
          f"{sampled['Class'].mean()*100:.3f}%")
    return sampled


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split into train/test sets and apply StandardScaler.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    scaler : fitted StandardScaler (save alongside the model)
    feature_names : list[str]
    """
    feature_names = [c for c in df.columns if c != "Class"]
    X = df[feature_names].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"Train fraud count: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    return X_train, X_test, y_train, y_test, scaler, feature_names


def smote_oversample(X_train: np.ndarray, y_train: np.ndarray):
    """
    Apply SMOTE oversampling to balance the training set.
    Requires imbalanced-learn to be installed.
    """
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — Train: {X_res.shape[0]:,} rows "
          f"| Fraud: {y_res.sum():,} ({y_res.mean()*100:.1f}%)")
    return X_res, y_res
