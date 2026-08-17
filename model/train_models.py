"""
train_models.py
----------------
Trains 5 classification models on the Optical Recognition of Handwritten
Digits dataset (UCI ML Repository, accessed via sklearn.datasets.load_digits),
evaluates them, and saves:
  - Trained model objects (model/*.joblib)
  - A fitted StandardScaler (model/scaler.joblib) for scale-sensitive models
  - test_data.csv (held-out test set, features + true label) at project root
  - metrics_comparison.csv (the comparison table used in README.md)

Run:
    python model/train_models.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
data = load_digits()
X = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])])
y = pd.Series(data.target, name="label")

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features, "
      f"{y.nunique()} classes")

# ---------------------------------------------------------------------------
# 2. Train/test split (stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale-sensitive models (Logistic Regression, KNN) get scaled features.
# Tree-based / NB models use the raw features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    "kNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes": (GaussianNB(), False),
    "Random Forest (Ensemble)": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), False),
    "SVM": (SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), True),
}

results = []
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train.values
    Xte = X_test_scaled if needs_scaling else X_test.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
    except ValueError:
        auc = np.nan

    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    })

    # Save the fitted model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(os.path.join(ROOT, "model", f"{fname}.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"Trained {name}: acc={acc:.4f} auc={auc:.4f} f1={f1:.4f} mcc={mcc:.4f}")

# Save scaler (needed at inference time for LR / kNN / SVM)
with open(os.path.join(ROOT, "model", "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save which models need scaling (used by the Streamlit app)
with open(os.path.join(ROOT, "model", "model_meta.json"), "w") as f:
    json.dump({name: needs_scaling for name, (_, needs_scaling) in models.items()}, f, indent=2)

# ---------------------------------------------------------------------------
# 4. Save comparison table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(ROOT, "metrics_comparison.csv"), index=False)
print("\nComparison Table:\n", results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Save test data (features + true label) for the Streamlit app upload
# ---------------------------------------------------------------------------
test_df = X_test.copy()
test_df["label"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"\nSaved test_data.csv with {len(test_df)} rows to project root.")
