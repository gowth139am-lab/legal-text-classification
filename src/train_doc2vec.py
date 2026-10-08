"""Train a Doc2Vec representation followed by Logistic Regression."""

import argparse

import joblib
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data import PROJECT_ROOT, tokenize, training_frame


def vectors(model: Doc2Vec, documents: list[list[str]]) -> np.ndarray:
    return np.array([model.infer_vector(tokens) for tokens in documents])


def train(target: str = "issue_area", limit: int | None = None) -> dict[str, float]:
    data = training_frame(target, limit)
    train_texts, test_texts, y_train, y_test = train_test_split(
        data["text"].tolist(),
        data[target].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=data[target],
    )
    train_tokens = [tokenize(text) for text in train_texts]
    test_tokens = [tokenize(text) for text in test_texts]
    documents = [
        TaggedDocument(words=tokens, tags=[str(index)])
        for index, tokens in enumerate(train_tokens)
    ]
    model = Doc2Vec(
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=20,
        dm=1,
        seed=42,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    classifier = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    classifier.fit(vectors(model, train_tokens), y_train)
    predictions = classifier.predict(vectors(model, test_tokens))
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
    }
    model_dir = PROJECT_ROOT / "models"
    model.save(str(model_dir / f"doc2vec_{target}.model"))
    joblib.dump(classifier, model_dir / f"doc2vec_classifier_{target}.joblib")
    (PROJECT_ROOT / "results" / f"doc2vec_{target}.txt").write_text(
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
