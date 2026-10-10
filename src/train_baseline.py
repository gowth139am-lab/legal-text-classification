"""Train and evaluate the TF-IDF plus Logistic Regression baseline."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.preprocess import ISSUE_AREA_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
METRICS_PATH = PROJECT_ROOT / "results" / "baseline_metrics.json"


def train_baseline() -> dict[str, float | str]:
    """Train the baseline model, print evaluation results, and save outputs."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    x_train = train["clean_text"].fillna("")
    y_train = train["issue_area"]
    x_test = test["clean_text"].fillna("")
    y_test = test["issue_area"]

    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(x_train, y_train)
    majority_predictions = majority_model.predict(x_test)
    majority_accuracy = accuracy_score(y_test, majority_predictions)
    majority_macro_f1 = f1_score(
        y_test, majority_predictions, average="macro", zero_division=0
    )
    print("Majority-class baseline")
    print(f"Accuracy: {majority_accuracy:.4f}")
    print(f"Macro-F1: {majority_macro_f1:.4f}")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

    labels = sorted(y_test.unique())
    target_names = [
        ISSUE_AREA_NAMES.get(int(label), str(label)) for label in labels
    ]
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    print("\nTF-IDF + Logistic Regression")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model": "tfidf_logistic_regression",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "majority_accuracy": float(majority_accuracy),
        "majority_macro_f1": float(majority_macro_f1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train_baseline()
