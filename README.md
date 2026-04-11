# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a custom-built Random Forest and a from-scratch SVM implementation. Trained on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | **0.9700** |
| False Positives | 3 |
| Top Fraud Predictor | V14 |
| Training Samples | 50,000 |

---

## Project Structure

```
credit-card-fraud-detection/
│
├── src/
│   ├── decision_tree.py     # Decision tree built from scratch (NumPy)
│   ├── random_forest.py     # Random Forest built from scratch (NumPy)
│   ├── svm.py               # SVM built from scratch (sub-gradient descent)
│   ├── preprocess.py        # Data loading, scaling, SMOTE oversampling
│   └── visualize.py         # Confusion matrix, ROC curve, feature importance
│
├── static/
│   └── app.html             # Interactive fraud detection UI (single + batch CSV)
│
├── data/
│   └── creditcard.csv       # Dataset (download from Kaggle — not included)
│
├── model/                   # Saved artifacts after running train.py (git-ignored)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── outputs/                 # Generated visualizations (git-ignored)
│
├── train.py                 # End-to-end training pipeline
├── predict.py               # Load model and predict on new transactions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/PeterBertels/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder:
```
data/creditcard.csv
```

---

## Usage

### Train the model
```bash
python train.py
```
Trains a Random Forest on 50,000 sampled transactions with SMOTE oversampling. Saves the model to `model/` and visualizations to `outputs/`.

### Predict on a new transaction
```bash
python predict.py
```
Or import in your own script:
```python
from predict import predict

result = predict([0.0, -1.36, -0.07, 2.54, 1.38, ...])  # 30 features
print(result)
# {'is_fraud': False, 'label': 'Legitimate', 'confidence': 94.2}
```

### Interactive UI
Open `static/app.html` in any browser. No server required.
- **Single Transaction** — enter 30 feature values manually or load a fraud example
- **Batch CSV Upload** — upload a CSV with the same 30-column format; get per-row predictions, summary stats, and an exportable results CSV

---

## Key Concepts

**SMOTE** — Synthetic Minority Oversampling Technique. Generates synthetic fraud samples during training so the model doesn't just learn to predict "legitimate" every time.

**Random Forest** — An ensemble of decision trees trained on bootstrap samples of the data. Each tree votes; the majority wins. Built from scratch in `src/random_forest.py`.

**SVM (from scratch)** — Soft-margin linear SVM trained via sub-gradient descent on the hinge loss objective. Implemented without scikit-learn in `src/svm.py`.

**ROC-AUC** — A threshold-independent metric suited to imbalanced classification. A score of 0.97 means the model can correctly rank fraud above legitimate transactions 97% of the time.

---

## Visualizations

All charts are saved to `outputs/` after running `train.py`:

| File | Description |
|---|---|
| `class_distribution.png` | Fraud vs. legitimate counts before SMOTE |
| `feature_distributions.png` | Histograms of V1, V2, V3, V14 |
| `confusion_matrix.png` | True/false positives and negatives |
| `roc_curve.png` | ROC curve with AUC annotation |
| `feature_importance.png` | Top 15 features ranked by importance |

---

## Author

Peter Bertels — Computer Science Student, Carleton University
