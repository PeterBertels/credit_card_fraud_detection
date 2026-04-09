import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the credit card fraud dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data: scale features and split."""
    # Sample a subset of the data for faster testing
    df = df.sample(n=10000, random_state=42)  # Sample 10,000 rows
    
    # Separate features and target
    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test