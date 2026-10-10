"""Clean Supreme Court opinions and create stratified train/test splits."""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ISSUE_AREA_NAMES = {
    1: "Criminal Procedure",
    2: "Civil Rights",
    3: "First Amendment",
    4: "Due Process",
    5: "Privacy",
    6: "Attorneys",
    7: "Unions",
    8: "Economic Activity",
    9: "Judicial Power",
    10: "Federalism",
    11: "Interstate Relations",
    12: "Federal Taxation",
    13: "Miscellaneous",
    14: "Private Action",
}

PROCESSED_DIR = Path("data/processed")
INPUT_PATH = PROCESSED_DIR / "cases.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


def clean_text(text: str) -> str:
    """Lowercase text, keep alphabetic characters, and normalize whitespace."""
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def make_splits(df: pd.DataFrame, label_col: str = "issue_area") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean data, remove unusable classes, and save stratified train/test splits."""
    if "text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")
    if label_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain the label column '{label_col}'.")

    working = df.copy()
    before_rows = len(working)
    working = working[working["text"].notna()]
    working = working[working["text"].astype(str).str.strip().ne("")]

    labels = working[label_col].astype("string").str.strip()
    working = working[working[label_col].notna() & labels.ne("-1")].copy()

    class_counts = working[label_col].value_counts()
    small_classes = class_counts[class_counts < 5]
    if not small_classes.empty:
        print("Dropping classes with fewer than 5 samples:")
        for class_value, count in small_classes.items():
            try:
                class_name = ISSUE_AREA_NAMES.get(int(class_value), str(class_value))
            except (TypeError, ValueError):
                class_name = str(class_value)
            print(f"  {class_value} ({class_name}): {count} rows")
        working = working[~working[label_col].isin(small_classes.index)].copy()
        print(f"Rows removed for small classes: {int(small_classes.sum())}")
    else:
        print("No classes with fewer than 5 samples were found.")

    working["clean_text"] = working["text"].map(clean_text)
    empty_clean_text = working["clean_text"].eq("")
    if empty_clean_text.any():
        working = working[~empty_clean_text].copy()
        print(f"Rows removed after cleaning to empty text: {int(empty_clean_text.sum())}")

    if working.empty:
        raise ValueError("No usable rows remain after preprocessing.")

    train, test = train_test_split(
        working,
        test_size=0.2,
        random_state=42,
        stratify=working[label_col],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Rows removed before class filtering and cleaning: {before_rows - len(working)}")
    print("Final train class counts:")
    print(train[label_col].value_counts().sort_index().to_string())
    print("Final test class counts:")
    print(test[label_col].value_counts().sort_index().to_string())
    return train, test


if __name__ == "__main__":
    make_splits(pd.read_csv(INPUT_PATH))
