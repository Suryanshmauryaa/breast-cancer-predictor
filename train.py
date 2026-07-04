"""
Breast Cancer Diagnosis Predictor
Loads the Wisconsin Breast Cancer dataset, does quick EDA, trains and
compares 3 classifiers, and saves the best one (by F1-score) to disk
along with the scaler and evaluation plots.

Run:
    python train.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df["target"] = data.target  # 0 = malignant, 1 = benign
    return df, data.target_names


def eda(df):
    print("Shape:", df.shape)
    print("\nClass balance:\n", df["target"].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df)
    plt.title("Class Distribution (0=Malignant, 1=Benign)")
    plt.savefig("plots/class_distribution.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)["target"].sort_values()
    top_corr = pd.concat([corr.head(6), corr.tail(6)])
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title("Features Most Correlated with Diagnosis")
    plt.savefig("plots/feature_correlation.png", bbox_inches="tight")
    plt.close()


def train_and_compare(X_train, X_test, y_train, y_test):
    models = {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    results = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
        }
        fitted[name] = model
        print(f"\n{name}:")
        for k, v in results[name].items():
            print(f"  {k}: {v}")

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    print(f"\nBest model by F1-score: {best_name}")
    return best_name, fitted[best_name], results


def save_confusion_matrix(model, X_test, y_test, target_names):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Best Model")
    plt.savefig("plots/confusion_matrix.png", bbox_inches="tight")
    plt.close()


def save_roc_curve(model, X_test, y_test):
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve - Best Model")
    plt.savefig("plots/roc_curve.png", bbox_inches="tight")
    plt.close()


def main():
    df, target_names = load_data()
    eda(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_name, best_model, all_results = train_and_compare(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    save_confusion_matrix(best_model, X_test_scaled, y_test, target_names)
    save_roc_curve(best_model, X_test_scaled, y_test)

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    with open("models/results.json", "w") as f:
        json.dump({"best_model": best_name, "all_results": all_results}, f, indent=2)

    print(f"\nSaved best model ({best_name}) and scaler to /models")
    print("Saved plots to /plots")


if __name__ == "__main__":
    main()
