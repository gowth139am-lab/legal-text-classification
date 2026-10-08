"""Train the TF-IDF + Logistic Regression baseline."""

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data import PROJECT_ROOT, training_frame


def train(target: str = "issue_area", limit: int | None = None) -> dict[str, float]:
    """Train, evaluate, and save a baseline model."""
    data = training_frame(target, limit)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data[target],
        test_size=0.2,
        random_state=42,
        stratify=data[target],
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
    }
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    joblib.dump(model, output_dir / f"baseline_{target}.joblib")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"baseline_{target}.txt").write_text(
        f"accuracy: {metrics['accuracy']:.4f}\nmacro_f1: {metrics['macro_f1']:.4f}\n",
        encoding="utf-8",
    )
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["issue_area", "issue"], default="issue_area")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    train(args.target, args.limit)
