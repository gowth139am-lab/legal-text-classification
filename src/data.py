"""Download and prepare the textacy Supreme Court dataset."""

from pathlib import Path
import re
from typing import Optional

import pandas as pd
from textacy.datasets import SupremeCourt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "supreme_court"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_FILE = PROCESSED_DATA_DIR / "supreme_court.csv"


def download_dataset() -> SupremeCourt:
    """Download the dataset into data/raw and return its reader."""
    dataset = SupremeCourt(RAW_DATA_DIR)
    dataset.download()
    return dataset


def load_dataset(
    limit: Optional[int] = None,
    min_len: int = 500,
    force_download: bool = False,
) -> pd.DataFrame:
    """Return opinion text and both supported labels as a DataFrame."""
    if DATA_FILE.exists() and not force_download:
        frame = pd.read_csv(DATA_FILE)
        return frame if limit is None else frame.head(limit)

    dataset = download_dataset()
    rows = []
    # Cache the complete labeled dataset so a quick smoke-test limit does not
    # permanently replace the full training data.
    for record in dataset.records(min_len=min_len):
        issue_area = record.meta.get("issue_area")
        issue = record.meta.get("issue")
        if issue_area in (None, -1) or issue in (None, "", "-1"):
            continue
        rows.append(
            {
                "text": record.text,
                "issue_area": str(issue_area),
                "issue": str(issue),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No labeled opinions were found in the downloaded dataset.")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_FILE, index=False)
    return frame if limit is None else frame.head(limit)


def tokenize(text: str) -> list[str]:
    """Create simple lowercase word tokens for gensim models."""
    return re.findall(r"[a-z]{2,}", text.lower())


def training_frame(target: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Prepare a stratifiable frame and optionally take a reproducible subset."""
    frame = load_dataset()
    counts = frame[target].value_counts()
    frame = frame[frame[target].isin(counts[counts >= 2].index)].reset_index(drop=True)
    if limit is not None and limit < len(frame):
        minimum = len(counts[counts >= 2]) * 2
        if limit < minimum:
            raise ValueError(
                f"limit must be at least {minimum} for a stratified split of {target}."
            )
        selected = []
        for _, group in frame.groupby(target):
            selected.extend(group.sample(n=2, random_state=42).index.tolist())
        remaining = frame.drop(index=selected)
        extra = limit - len(selected)
        if extra:
            selected.extend(
                remaining.sample(n=extra, random_state=42).index.tolist()
            )
        frame = frame.loc[selected]
        frame = frame.reset_index(drop=True)
    return frame
