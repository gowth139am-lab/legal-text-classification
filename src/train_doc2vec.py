"""Train and evaluate a Doc2Vec plus Logistic Regression classifier."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.preprocess import ISSUE_AREA_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "doc2vec.model"
METRICS_PATH = PROJECT_ROOT / "results" / "doc2vec_metrics.json"


def _tokens(text: str) -> list[str]:
    """Split cleaned text into the tokens used by Doc2Vec."""
    return str(text).split()


def train_doc2vec() -> dict[str, float | str]:
    """Train Doc2Vec and Logistic Regression, then save the outputs."""
    print("Loading train and test data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_tokens = [_tokens(text) for text in train["clean_text"].fillna("")]
    test_tokens = [_tokens(text) for text in test["clean_text"].fillna("")]
    y_train = train["issue_area"]
    y_test = test["issue_area"]

    print(f"Loaded {len(train)} training rows and {len(test)} test rows.")
    print("Creating TaggedDocument objects from the training set...")
    tagged_documents = [
        TaggedDocument(words=tokens, tags=[str(index)])
        for index, tokens in enumerate(train_tokens)
    ]

    print("Building the Doc2Vec vocabulary...")
    model = Doc2Vec(
        vector_size=100,
        window=5,
        min_count=5,
        epochs=20,
        workers=4,
        seed=42,
    )
    model.build_vocab(tagged_documents)
    print(
        f"Training Doc2Vec for {model.epochs} epochs "
        f"with {len(model.wv)} vocabulary terms..."
    )
    model.train(
        tagged_documents,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    print("Getting training vectors from model.dv...")
    train_vectors = np.array([model.dv[str(index)] for index in range(len(train))])

    print("Inferring test vectors...")
    test_vectors = np.array(
        [model.infer_vector(tokens, epochs=20) for tokens in test_tokens]
    )

    print("Training Logistic Regression...")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(train_vectors, y_train)

    print("Evaluating the classifier...")
    predictions = classifier.predict(test_vectors)
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
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))

    metrics = {
        "model": "doc2vec_logistic_regression",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Doc2Vec model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train_doc2vec()
