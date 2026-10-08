"""Download and save the textacy Supreme Court dataset."""

from pathlib import Path
from typing import Any

import pandas as pd
from textacy.datasets import SupremeCourt


DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/cases.csv")


def _get_year(metadata: dict[str, Any]) -> Any:
    """Return a year from metadata when one is available."""
    if metadata.get("year") is not None:
        return metadata["year"]

    decision_date = metadata.get("decision_date")
    if decision_date:
        return str(decision_date)[:4]
    return None


def load_dataframe() -> pd.DataFrame:
    """Download the dataset if needed and save all records as a DataFrame."""
    dataset = SupremeCourt(data_dir=DATA_DIR)
    dataset_path = Path(dataset.filepath) if dataset.filepath else None

    if dataset_path is None or not dataset_path.exists():
        try:
            print("Supreme Court dataset not found. Downloading it now...")
            dataset.download()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to download the Supreme Court dataset to {DATA_DIR}: {exc}"
            ) from exc

    rows = []
    try:
        for record in dataset.records():
            metadata = record.meta
            rows.append(
                {
                    "text": record.text,
                    "issue": metadata.get("issue"),
                    "issue_area": metadata.get("issue_area"),
                    "case_name": metadata.get("case_name"),
                    "year": _get_year(metadata),
                }
            )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read Supreme Court records from {dataset_path}: {exc}"
        ) from exc

    dataframe = pd.DataFrame(
        rows,
        columns=["text", "issue", "issue_area", "case_name", "year"],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset size: {len(dataframe)} rows")
    print("Missing values per column:")
    print(dataframe.isna().sum().to_string())
    print(f"Unique issue_area values: {dataframe['issue_area'].nunique()}")
    print(f"Unique issue values: {dataframe['issue'].nunique()}")
    print(f"Saved DataFrame to {OUTPUT_PATH}")
    return dataframe


if __name__ == "__main__":
    load_dataframe()
