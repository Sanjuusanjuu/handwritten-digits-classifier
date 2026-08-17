# ML Assignment 2 — Handwritten Digits Classification

**Course:** Machine Learning — M.Tech (AIML / DSE), BITS Pilani WILP
**Name:** SANJANA P | **BITS ID:** 2025DA04262

---

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy an end-to-end
machine learning classification pipeline. Six evaluation metrics (Accuracy,
AUC, Precision, Recall, F1, MCC) are computed for six different
classification algorithms trained on the same dataset, and the results are
exposed through an interactive Streamlit web application deployed on
Streamlit Community Cloud, where a user can upload test data, pick a model,
and view its performance (including a confusion matrix / classification
report) live.

This is framed as a **multi-class digit recognition problem**: given an 8×8
grayscale image of a handwritten digit (represented as 64 pixel-intensity
features), predict which digit (0–9) it represents.

## b. Dataset Description

- **Name:** Optical Recognition of Handwritten Digits Data Set
- **Source:** UCI Machine Learning Repository (bundled and accessed offline
  via `sklearn.datasets.load_digits`, which loads the standard UCI
  `optdigits` data)
- **Instances:** 1,797
- **Features:** 64 (pixel intensity values 0–16 from an 8×8 image grid,
  flattened — i.e., `pixel_0` … `pixel_63`)
- **Target:** `label` — digit class, 0 through 9 (10-class classification)
- **Train/Test split:** 80% / 20%, stratified by class (`random_state=42`)
- **Class balance:** Roughly uniform across the 10 digit classes (~180
  samples each)

This satisfies the assignment's minimum requirements of **≥12 features**
(64 present) and **≥500 instances** (1,797 present).

> Note: The dataset is loaded from scikit-learn's bundled copy of the UCI
> `optdigits` dataset rather than downloaded fresh, since the assignment is
> performed on the BITS Virtual Lab, which has restricted internet access.
> This is the same public UCI dataset referenced at:
> https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

## c. GitHub Repository Link

https://github.com/Sanjuusanjuu/handwritten-digits-classifier.git

Repository structure:

```
project-folder/
│-- app.py                     # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out test set (features + true label)
│-- metrics_comparison.csv     # generated comparison table
│-- model/
    │-- train_models.py        # trains all 6 models + saves artifacts
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest_ensemble.pkl
    │-- svm.pkl
    │-- scaler.pkl
    │-- model_meta.json
```

## d. Models Used

Six classification models were trained on identical train/test splits of
the same dataset. All metrics below are computed on the held-out 20% test
set (360 samples), using **weighted averaging** for Precision/Recall/F1
(appropriate for the 10-class problem) and **one-vs-rest weighted AUC** for
the multi-class ROC-AUC score.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9722 | 0.9991 | 0.9724 | 0.9722 | 0.9722 | 0.9692 |
| Decision Tree | 0.8222 | 0.9012 | 0.8234 | 0.8222 | 0.8214 | 0.8028 |
| kNN | 0.9639 | 0.9951 | 0.9648 | 0.9639 | 0.9636 | 0.9600 |
| Naive Bayes | 0.8111 | 0.9707 | 0.8480 | 0.8111 | 0.8151 | 0.7940 |
| Random Forest (Ensemble) | 0.9639 | 0.9992 | 0.9644 | 0.9639 | 0.9636 | 0.9600 |
| SVM | 0.9750 | 0.9995 | 0.9759 | 0.9750 | 0.9749 | 0.9723 |

*(These exact figures are reproduced by running `python model/train_models.py`
with `random_state=42`; they are also saved to `metrics_comparison.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Surprisingly a top performer here — the digit pixel features are close to linearly separable after standardization, so a simple linear decision boundary per class (one-vs-rest) generalizes very well. High accuracy, F1, and MCC, with near-perfect AUC. Training and inference are also the fastest of all six models. |
| Decision Tree | Clearly the weakest model. A single tree overfits the training pixels and produces jagged, axis-aligned decision boundaries that don't generalize well to a continuous-valued image dataset — accuracy and MCC are 10–15 points below every other model. |
| kNN | Performs very well (96.4% accuracy) because digit images from the same class tend to be genuinely close to each other in raw pixel space, so distance-based matching works effectively once features are scaled. Slightly behind Logistic Regression and Random Forest on AUC. |
| Naive Bayes | Second-weakest model. Gaussian Naive Bayes assumes each pixel feature is independent given the class, which is a poor assumption for image data (neighboring pixels are highly correlated), capping its accuracy around 81%. Its AUC is still reasonably high, showing it ranks classes sensibly even when its hard predictions are wrong. |
| Random Forest (Ensemble) | Strong and robust — matches kNN on accuracy/F1/MCC and edges out every other model (including Logistic Regression) on AUC. By averaging many de-correlated trees it fixes the single Decision Tree's overfitting problem while keeping good class-probability calibration. |
| SVM | Best overall performer on this dataset — achieves the highest accuracy (97.5%), AUC (0.9995), Precision, F1, and MCC of all six models. The RBF kernel effectively captures non-linear boundaries in the 64-dimensional pixel space, and the maximum-margin objective produces well-calibrated class separations. Requires standardization but rewards it with superior generalization. |
| **Overall Winner for your dataset?** | **SVM** (RBF kernel) — highest on every metric (accuracy 0.9750, AUC 0.9995, MCC 0.9723). Logistic Regression is a close second and much faster to train; Random Forest is the best tree-based option. Decision Tree and Naive Bayes are clearly behind. |

## Streamlit App Features

The deployed app (`app.py`) implements:

- **(a) Dataset upload (CSV):** upload `test_data.csv` (or any CSV with the
  same `pixel_*` feature columns, optionally with a `label` column).
- **(b) Model selection dropdown:** choose from the 6 trained models.
- **(c) Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1,
  MCC shown live for the uploaded data (when true labels are present).
- **(d) Confusion matrix / classification report:** heatmap + full
  per-class `classification_report`.
- A bonus section showing the full 6-model comparison table computed
  during training.

## Live Streamlit App Link

https://handwritten-digits-classifier-wbzchm3b2jncxxgdbtl7yc.streamlit.app/

## How to Reproduce Locally

```bash
pip install -r requirements.txt
python model/train_models.py     # trains all 6 models, writes test_data.csv
                                  # and metrics_comparison.csv
streamlit run app.py
```


