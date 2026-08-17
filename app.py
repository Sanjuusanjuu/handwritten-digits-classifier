"""
Streamlit App - Handwritten Digits Classification Demo
--------------------------------------------------------
Assignment 2 - Machine Learning (M.Tech AIML/DSE), BITS Pilani WILP
Author: SANJANA P | BITS ID: 2025DA04262

Features implemented (per assignment spec):
  a. Dataset upload option (CSV)              -> sidebar file uploader
  b. Model selection dropdown                 -> sidebar selectbox
  c. Display of evaluation metrics            -> metric cards + bar chart
  d. Confusion matrix / classification report -> heatmap + text report
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# -----------------------------------------------------------------------
# Page config & custom CSS
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Digits Classifier | BITS ML Assignment 2",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Main background — deep navy */
    .stApp { background-color: #0d1b2a; color: #e8eaf6; }

    /* All default text */
    .stApp p, .stApp span, .stApp div, .stApp label { color: #e8eaf6; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1b2a3b; }
    section[data-testid="stSidebar"] * { color: #cfd8dc !important; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #1e3a5f;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.4);
        border-left: 4px solid #5c8dd6;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div { color: #ffffff !important; }

    /* Section headers */
    h1, h2, h3 { color: #90caf9 !important; }
    h2 { border-bottom: 2px solid #5c8dd6; padding-bottom: 4px; }

    /* Tab bar */
    button[data-baseweb="tab"] { color: #90caf9 !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 3px solid #5c8dd6 !important;
    }

    /* Info / warning boxes */
    div[data-testid="stAlert"] { background-color: #1e3a5f; border-radius: 8px; }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Code block */
    pre { background-color: #1e3a5f !important; color: #e8eaf6 !important; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "SVM": "svm.pkl",
}

METRIC_LABELS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@st.cache_resource
def load_artifacts():
    models = {}
    for name, fname in MODEL_FILES.items():
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            models[name] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        needs_scaling = json.load(f)
    return models, scaler, needs_scaling


models, scaler, needs_scaling = load_artifacts()

# -----------------------------------------------------------------------
# Sidebar — controls
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔢 Digits Classifier")
    st.markdown("**BITS Pilani WILP — ML Assignment 2**")
    st.markdown("---")

    st.markdown("### a. Upload Test Data")
    uploaded = st.file_uploader(
        "Upload CSV (test_data.csv or compatible)",
        type=["csv"],
        help="Must have pixel_0 … pixel_63 columns. Include a 'label' column for evaluation.",
    )

    st.markdown("### b. Select Model")
    model_name = st.selectbox(
        "Choose a classification model",
        list(models.keys()),
        help="All 6 models were trained on the same UCI digits dataset.",
    )

    st.markdown("---")
    st.markdown(
        "**Dataset:** UCI Optical Handwritten Digits  \n"
        "**Instances:** 1,797 | **Features:** 64  \n"
        "**Classes:** 10 (digits 0–9)"
    )

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(os.path.join(HERE, "test_data.csv"))

has_labels = "label" in df.columns
feature_cols = [c for c in df.columns if c != "label"]

model = models[model_name]
X = df[feature_cols].values
X_input = scaler.transform(X) if needs_scaling.get(model_name, False) else X

y_pred = model.predict(X_input)
y_proba = model.predict_proba(X_input)

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("Handwritten Digits — Multi-Model Classification Dashboard")
st.caption(
    f"Showing results for **{model_name}** on **{df.shape[0]} samples** "
    f"({'with' if has_labels else 'without'} true labels)"
)

if not uploaded:
    st.info("No file uploaded — using bundled `test_data.csv`. Upload your own CSV via the sidebar.")

# -----------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Metrics & Report", "🟦 Confusion Matrix", "📈 Model Comparison", "🔍 Data Preview"]
)

# ===================== TAB 1: Metrics & Report =====================
with tab1:
    if has_labels:
        y_true = df["label"].values
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        y_true_bin = label_binarize(y_true, classes=classes)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc  = matthews_corrcoef(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
        except ValueError:
            auc = float("nan")

        # c. Evaluation metrics
        st.subheader(f"c. Evaluation Metrics — {model_name}")
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        c1.metric("Accuracy",  f"{acc:.4f}")
        c2.metric("AUC Score", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall",    f"{rec:.4f}")
        c5.metric("F1 Score",  f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")

        # Metrics bar chart
        st.markdown("#### Metric Scores at a Glance")
        metric_vals = [acc, auc, prec, rec, f1, mcc]
        fig_bar, ax_bar = plt.subplots(figsize=(6, 2.5), facecolor="#0d1b2a")
        ax_bar.set_facecolor("#0d1b2a")
        colors = ["#5c8dd6" if v >= 0.95 else "#7986cb" if v >= 0.85 else "#455a64"
                  for v in metric_vals]
        bars = ax_bar.barh(METRIC_LABELS, metric_vals, color=colors)
        ax_bar.set_xlim(0, 1.1)
        ax_bar.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, color="#e8eaf6")
        ax_bar.set_xlabel("Score", color="#e8eaf6")
        ax_bar.set_title(f"{model_name} — All Metrics", fontsize=11, color="#90caf9")
        ax_bar.tick_params(colors="#e8eaf6")
        ax_bar.spines[:].set_color("#2a4a6b")
        ax_bar.invert_yaxis()
        fig_bar.tight_layout()
        st.pyplot(fig_bar)

        # d. Classification report
        st.subheader("d. Classification Report")
        report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
        st.code(report, language=None)

    else:
        st.warning(
            "No `label` column found — upload a CSV with true labels to see metrics. "
            "Showing raw predictions below."
        )
        st.dataframe(
            pd.DataFrame({"Sample Index": range(len(y_pred)), "Predicted Label": y_pred}),
            use_container_width=True,
        )

# ===================== TAB 2: Confusion Matrix =====================
with tab2:
    if has_labels:
        st.subheader(f"Confusion Matrix — {model_name}")
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3), facecolor="#0d1b2a")
        ax_cm.set_facecolor("#0d1b2a")
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax_cm,
            linewidths=0.5, linecolor="#0d1b2a",
        )
        ax_cm.set_xlabel("Predicted Label", fontsize=11, color="#e8eaf6")
        ax_cm.set_ylabel("True Label", fontsize=11, color="#e8eaf6")
        ax_cm.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12, color="#90caf9")
        ax_cm.tick_params(colors="#e8eaf6")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

        # Per-class accuracy bar
        st.markdown("#### Per-Class Recall (Sensitivity)")
        per_class_recall = cm.diagonal() / cm.sum(axis=1)
        fig_pcr, ax_pcr = plt.subplots(figsize=(6, 2.5), facecolor="#0d1b2a")
        ax_pcr.set_facecolor("#0d1b2a")
        ax_pcr.bar(classes, per_class_recall, color="#5c8dd6", alpha=0.85)
        ax_pcr.set_ylim(0, 1.15)
        ax_pcr.set_xticks(classes)
        ax_pcr.set_xlabel("Digit Class", color="#e8eaf6")
        ax_pcr.set_ylabel("Recall", color="#e8eaf6")
        ax_pcr.set_title("Recall per Digit Class", color="#90caf9")
        ax_pcr.tick_params(colors="#e8eaf6")
        ax_pcr.spines[:].set_color("#2a4a6b")
        for i, v in enumerate(per_class_recall):
            ax_pcr.text(classes[i], v + 0.03, f"{v:.2f}", ha="center", fontsize=8, color="#e8eaf6")
        fig_pcr.tight_layout()
        st.pyplot(fig_pcr)
    else:
        st.info("Upload a CSV with a `label` column to view the confusion matrix.")

# ===================== TAB 3: Model Comparison =====================
with tab3:
    st.subheader("All-Model Comparison (pre-computed on held-out test set)")
    try:
        comp_df = pd.read_csv(os.path.join(HERE, "metrics_comparison.csv"))
        st.dataframe(
            comp_df.style.highlight_max(
                subset=METRIC_LABELS, color="#c8e6c9"
            ).highlight_min(
                subset=METRIC_LABELS, color="#ffcdd2"
            ).format({m: "{:.4f}" for m in METRIC_LABELS}),
            use_container_width=True,
        )

        # Grouped bar chart across models
        st.markdown("#### Side-by-Side Metric Comparison")
        selected_metrics = st.multiselect(
            "Select metrics to plot",
            METRIC_LABELS,
            default=["Accuracy", "F1", "MCC"],
        )
        if selected_metrics:
            fig_cmp, ax_cmp = plt.subplots(figsize=(8, 3), facecolor="#0d1b2a")
            ax_cmp.set_facecolor("#0d1b2a")
            x = np.arange(len(comp_df))
            width = 0.8 / len(selected_metrics)
            palette = ["#5c8dd6", "#e91e63", "#26a69a", "#ff9800", "#ab47bc", "#8d6e63"]
            for i, metric in enumerate(selected_metrics):
                offset = (i - len(selected_metrics) / 2 + 0.5) * width
                ax_cmp.bar(x + offset, comp_df[metric], width, label=metric,
                           color=palette[i % len(palette)], alpha=0.85)
            ax_cmp.set_xticks(x)
            ax_cmp.set_xticklabels(comp_df["ML Model Name"], rotation=15, ha="right",
                                   fontsize=9, color="#e8eaf6")
            ax_cmp.set_ylim(0, 1.15)
            ax_cmp.set_ylabel("Score", color="#e8eaf6")
            ax_cmp.set_title("Model Comparison by Selected Metrics", color="#90caf9")
            ax_cmp.tick_params(colors="#e8eaf6")
            ax_cmp.spines[:].set_color("#2a4a6b")
            ax_cmp.legend(loc="lower right", facecolor="#1e3a5f", labelcolor="#e8eaf6")
            fig_cmp.tight_layout()
            st.pyplot(fig_cmp)
    except FileNotFoundError:
        st.warning("Run `python model/train_models.py` to generate `metrics_comparison.csv`.")

# ===================== TAB 4: Data Preview =====================
with tab4:
    st.subheader("Uploaded / Default Test Data Preview")
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("#### Prediction Distribution")
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    fig_dist, ax_dist = plt.subplots(figsize=(6, 2.5), facecolor="#0d1b2a")
    ax_dist.set_facecolor("#0d1b2a")
    ax_dist.bar(pred_counts.index, pred_counts.values, color="#5c8dd6", alpha=0.85)
    ax_dist.set_xlabel("Predicted Digit Class", color="#e8eaf6")
    ax_dist.set_ylabel("Count", color="#e8eaf6")
    ax_dist.set_title(f"Prediction Distribution — {model_name}", color="#90caf9")
    ax_dist.set_xticks(pred_counts.index)
    ax_dist.tick_params(colors="#e8eaf6")
    ax_dist.spines[:].set_color("#2a4a6b")
    fig_dist.tight_layout()
    st.pyplot(fig_dist)

st.markdown("---")
st.caption("BITS Pilani WILP · M.Tech (AIML/DSE) · Machine Learning Assignment 2 · SANJANA P · 2025DA04262")
