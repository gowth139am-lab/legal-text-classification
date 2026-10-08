"""Find opinions similar to a query using a saved Doc2Vec model."""

import argparse

from gensim.models.doc2vec import Doc2Vec

from data import PROJECT_ROOT, load_dataset, tokenize


def find_similar(query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
    """Return (row number, similarity, text) tuples for the closest opinions."""
    model = Doc2Vec.load(str(PROJECT_ROOT / "models" / "doc2vec_issue_area.model"))
    data = load_dataset()
    query_vector = model.infer_vector(tokenize(query))
    similarities = model.dv.cosine_similarities(query_vector, model.dv.vectors)
    results = [
        (index, float(score), data.iloc[index]["text"])
        for index, score in enumerate(similarities)
        if index < len(data)
    ]
    return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    for index, score, text in find_similar(args.query, args.top_k):
        print(f"\n#{index} similarity={score:.3f}\n{text[:500]}...")
