"""Train and evaluate an LDA topic representation with Logistic Regression."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.preprocess import ISSUE_AREA_NAMES


NUM_TOPICS = 50
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lda.model"
DICTIONARY_PATH = PROJECT_ROOT / "models" / "lda_dictionary.dict"
METRICS_PATH = PROJECT_ROOT / "results" / "lda_metrics.json"


def _tokens(text: str) -> list[str]:
    """Split cleaned text and remove stopwords and short words."""
    return [
        word
        for word in str(text).split()
        if len(word) >= 3 and word not in STOPWORDS
    ]


def _topic_vectors(
    model: LdaModel, dictionary: Dictionary, documents: list[list[str]]
) -> np.ndarray:
    """Convert documents into dense topic-probability vectors."""
    vectors = np.zeros((len(documents), NUM_TOPICS), dtype=float)
    for row, tokens in enumerate(documents):
        for topic_id, probability in model.get_document_topics(
            dictionary.doc2bow(tokens),
            minimum_probability=0,
        ):
            vectors[row, topic_id] = probability
    return vectors


def train_lda() -> dict[str, float | str]:
    """Train LDA and Logistic Regression, then save the requested outputs."""
    print("Loading train and test data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Loaded {len(train)} training rows and {len(test)} test rows.")

    print("Tokenizing training and test documents...")
    train_tokens = [_tokens(text) for text in train["clean_text"].fillna("")]
    test_tokens = [_tokens(text) for text in test["clean_text"].fillna("")]

    print("Building the dictionary from the training set only...")
    dictionary = Dictionary(train_tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    train_corpus = [dictionary.doc2bow(tokens) for tokens in train_tokens]
    test_corpus = [dictionary.doc2bow(tokens) for tokens in test_tokens]
    print(f"Dictionary contains {len(dictionary)} terms.")

    print(f"Training LDA with {NUM_TOPICS} topics for 5 passes...")
    lda = LdaModel(
        corpus=train_corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=5,
        random_state=42,
    )

    print("Converting documents to dense topic-probability vectors...")
    train_vectors = _topic_vectors(lda, dictionary, train_tokens)
    test_vectors = _topic_vectors(lda, dictionary, test_tokens)

    print("Training Logistic Regression...")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(train_vectors, train["issue_area"])

    print("Evaluating the classifier...")
    predictions = classifier.predict(test_vectors)
    accuracy = accuracy_score(test["issue_area"], predictions)
    macro_f1 = f1_score(
        test["issue_area"],
        predictions,
        average="macro",
        zero_division=0,
    )

    labels = sorted(test["issue_area"].unique())
    target_names = [
        ISSUE_AREA_NAMES.get(int(label), str(label)) for label in labels
    ]
    report = classification_report(
        test["issue_area"],
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
    lda.save(str(MODEL_PATH))
    dictionary.save(str(DICTIONARY_PATH))

    metrics = {
        "model": "lda_logistic_regression",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved LDA model to {MODEL_PATH}")
    print(f"Saved dictionary to {DICTIONARY_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train_lda()
