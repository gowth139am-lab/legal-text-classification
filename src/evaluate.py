"""Compare saved models and create evaluation plots."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.preprocess import ISSUE_AREA_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


def load_metrics() -> pd.DataFrame:
    """Load available model metrics into a comparison table."""
    rows = []
    for filename in ("baseline_metrics.json", "lda_metrics.json", "doc2vec_metrics.json"):
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"Skipping missing metrics file: {path}")
            continue
        with path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        rows.append(
            {
                "model": metrics["model"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    return pd.DataFrame(rows, columns=["model", "accuracy", "macro_f1"])


def save_model_comparison(comparison: pd.DataFrame) -> None:
    """Save the comparison table and grouped metric chart."""
    comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print(comparison.to_string(index=False))

    chart_data = comparison.set_index("model")[["accuracy", "macro_f1"]]
    ax = chart_data.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(["Accuracy", "Macro-F1"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()


def save_baseline_confusion_matrix() -> None:
    """Predict the test set and save a row-normalized baseline heatmap."""
    model_path = MODELS_DIR / "baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {model_path}")

    print("Loading baseline model and test data...")
    model = joblib.load(model_path)
    test = pd.read_csv(TEST_PATH)
    predictions = model.predict(test["clean_text"].fillna(""))

    labels = sorted(set(test["issue_area"]) | set(predictions))
    class_names = [ISSUE_AREA_NAMES.get(int(label), str(label)) for label in labels]
    matrix = confusion_matrix(
        test["issue_area"],
        predictions,
        labels=labels,
        normalize="true",
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
    )
    plt.title("Baseline Confusion Matrix (Row-Normalized)")
    plt.xlabel("Predicted issue area")
    plt.ylabel("True issue area")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix_baseline.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix_baseline.png'}")


def main() -> None:
    """Create all requested evaluation outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison = load_metrics()
    save_model_comparison(comparison)
    save_baseline_confusion_matrix()


if __name__ == "__main__":
    main()
