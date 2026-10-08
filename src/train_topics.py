"""Train an LDA document representation followed by Logistic Regression."""

import argparse

import joblib
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data import PROJECT_ROOT, tokenize, training_frame


def document_topics(model: LdaModel, dictionary: Dictionary, documents: list[list[str]]) -> np.ndarray:
    """Convert tokenized documents into dense topic-probability vectors."""
    vectors = np.zeros((len(documents), model.num_topics), dtype=float)
    for row, tokens in enumerate(documents):
        for topic_id, probability in model.get_document_topics(
            dictionary.doc2bow(tokens), minimum_probability=0
        ):
            vectors[row, topic_id] = probability
    return vectors


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
    dictionary = Dictionary(train_tokens)
    dictionary.filter_extremes(no_below=2, no_above=0.95, keep_n=50000)
    lda = LdaModel(
        [dictionary.doc2bow(tokens) for tokens in train_tokens],
        num_topics=50,
        id2word=dictionary,
        passes=5,
        random_state=42,
    )
    classifier = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    classifier.fit(document_topics(lda, dictionary, train_tokens), y_train)
    predictions = classifier.predict(document_topics(lda, dictionary, test_tokens))
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
    }
    bundle = {"dictionary": dictionary, "lda": lda, "classifier": classifier}
    joblib.dump(bundle, PROJECT_ROOT / "models" / f"lda_{target}.joblib")
    (PROJECT_ROOT / "results" / f"lda_{target}.txt").write_text(
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
