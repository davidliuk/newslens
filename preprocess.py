import pandas as pd
from typing import List, Tuple


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Load a news headlines CSV and return (headlines, labels).

    Expected CSV columns: 'headline', 'news_source' (fox/nbc)
    Also accepts a pre-encoded 'label' column (0/1) if present.

    Returns:
        X: list of headline strings
        y: list of int labels  (fox=0, nbc=1)
    """
    df = pd.read_csv(path)

    if "headline" not in df.columns:
        raise ValueError(f"CSV at '{path}' must have a 'headline' column. Found: {list(df.columns)}")

    X = df["headline"].fillna("").tolist()

    if "label" in df.columns:
        y = df["label"].astype(int).tolist()
    elif "news_source" in df.columns:
        y = df["news_source"].map({"fox": 0, "nbc": 1}).astype(int).tolist()
    else:
        raise ValueError(f"CSV at '{path}' must have a 'news_source' or 'label' column.")

    return X, y
