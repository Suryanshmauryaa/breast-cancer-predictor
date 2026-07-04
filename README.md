# Breast Cancer Diagnosis Predictor

A machine learning pipeline that classifies breast tumors as **benign** or **malignant**
using the Wisconsin Breast Cancer diagnostic dataset (569 samples, 30 features).
Includes EDA, model comparison across 3 classifiers, and an interactive Streamlit
demo app.

## Results

Trained and compared 3 classifiers on an 80/20 stratified train-test split:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.9825 | 0.9861 | 0.9861 | **0.9861** | 0.9954 |
| SVM (RBF kernel) | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9932 |

Logistic Regression was selected as the best model (highest F1-score) and is the
one used in the demo app.

*(Numbers are reproducible — run `train.py` yourself, they'll come out the same
since the random seed is fixed.)*

## How it works

### Problem statement
This is a binary classification problem: given 30 diagnostic measurements taken
from a tumor's cell nuclei (radius, texture, perimeter, area, smoothness,
compactness, concavity, symmetry, fractal dimension — each as a mean, standard
error, and "worst" value), predict whether the tumor is **malignant** (cancerous)
or **benign** (non-cancerous).

### Pipeline

**1. Data loading**
The dataset (569 samples) is loaded via `sklearn.datasets.load_breast_cancer()`
directly into a pandas DataFrame — no external download required.

**2. Exploratory Data Analysis (EDA)**
Before modeling, the data is inspected to understand class balance (212
malignant vs. 357 benign) and which features correlate most strongly with
diagnosis. This avoids building a model blind to biases in the underlying data.

**3. Train-test split**
The data is split 80/20 with stratification (`stratify=y`), so both the
training and test sets preserve the same class ratio. The model is trained
only on the 80% split and evaluated on the untouched 20%, preventing
overfitting from being mistaken for real performance.

**4. Feature scaling**
The 30 features live on very different numeric scales (e.g. "area" is in the
hundreds, "smoothness" is a small decimal). `StandardScaler` transforms every
feature to zero mean and unit variance:

```
scaled_value = (value - mean) / standard_deviation
```

This keeps scale-sensitive models like Logistic Regression and SVM from being
skewed by features with naturally larger numeric ranges.

**5. Model training & comparison**
Three classifiers are trained on identical data and compared:

- **Logistic Regression** — a linear decision boundary; fast and interpretable.
- **Random Forest** — an ensemble of 200 decision trees voting together;
  captures non-linear patterns.
- **SVM (RBF kernel)** — finds a non-linear boundary well suited to data that
  isn't linearly separable.

**6. Evaluation**
Each model is scored on five metrics: accuracy, precision, recall, F1-score,
and ROC-AUC. The model with the highest **F1-score** is selected as the final
model — F1 balances precision and recall, which matters here since both false
positives and false negatives carry real cost in a diagnostic setting.

**7. Persistence & deployment**
The winning model, its scaler, and feature order are saved with `joblib` so
the Streamlit app can load them instantly and make predictions without
retraining.

### Metrics, explained

| Metric | What it measures |
|---|---|
| Accuracy | Overall % of correct predictions |
| Precision | Of all "benign" predictions, how many were truly benign (false-positive control) |
| Recall | Of all actual benign cases, how many were correctly identified (false-negative control — critical in diagnosis) |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | How well the model separates the two classes across all decision thresholds |

### Why F1-score over raw accuracy?
The dataset has a mild class imbalance (63% benign vs. 37% malignant), and in
a medical context a false negative (missing a malignant tumor) is far more
costly than a false positive. F1-score accounts for both precision and
recall rather than optimizing for accuracy alone, which can be misleading on
imbalanced data.

## Project structure

```
breast-cancer-predictor/
├── train.py              # EDA + trains & compares models, saves the best one
├── app.py                # Streamlit app for interactive predictions
├── requirements.txt
├── models/                # generated after running train.py
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── results.json
└── plots/                 # generated after running train.py
    ├── class_distribution.png
    ├── feature_correlation.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/breast-cancer-predictor.git
cd breast-cancer-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models (generates /models and /plots)
python train.py

# 4. Launch the interactive demo
streamlit run app.py
```

## Tech stack

- **scikit-learn** — model training, evaluation, scaling
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — EDA and evaluation visualizations
- **Streamlit** — interactive prediction demo
- **joblib** — model persistence

## Dataset

[Wisconsin Breast Cancer Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
loaded directly via `sklearn.datasets.load_breast_cancer` — no manual download needed.

## Limitations

- The dataset is relatively small (569 samples); a production system would
  need significantly more data.
- Evaluated on a single dataset — performance may vary on data collected from
  different equipment or patient populations.
- No hyperparameter tuning was performed (models use default/near-default
  settings); GridSearch or RandomizedSearch could likely improve results.
- This is a portfolio/learning project and is **not intended for real
  clinical use**.

## Author

Suryansh Maurya
LinkedIn - https://linkedin.com/in/suryanshmaurya 
GitHub - https://github.com/Suryanshmauryaa
