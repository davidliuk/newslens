import re
from typing import List, Tuple
from urllib.parse import unquote, urlparse

import pandas as pd


def _label_from_source_text(text: str):
    text = str(text).lower()
    if "foxnews" in text or "foxbusiness" in text:
        return 0
    if "nbcnews" in text or "msnbc" in text or "cnbc" in text or "today.com" in text:
        return 1
    return None


def _headline_from_url(url: str) -> str:
    path = unquote(urlparse(str(url)).path)
    segments = [seg for seg in path.split("/") if seg]
    if not segments:
        return ""

    def clean_segment(segment: str) -> List[str]:
        segment = segment.lower()
        segment = re.sub(r"\.(html?|print)$", "", segment)
        segment = re.sub(r"\b(?:rcna|ncna|bbna)\d+\b", "", segment)
        segment = re.sub(r"\b\d{4}\b", "", segment)
        segment = re.sub(r"[^a-z0-9]+", " ", segment)
        return [tok for tok in segment.split() if tok and not tok.isdigit()]

    drop = {"www", "foxnews", "nbcnews", "fox", "nbc", "news", "com"}

    cleaned_segments = [[word for word in clean_segment(seg) if word not in drop] for seg in segments]
    words = [word for segment_words in cleaned_segments for word in segment_words]
    return " ".join(words).strip()


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Load a news headlines CSV and return (headlines, labels).

    Expected CSV columns: 'headline' or 'url', plus labels when available.
    Labels can be in 'label' (0/1) or 'news_source' (fox/nbc). For the
    leaderboard's URL-only files, extract readable headline text from the URL.

    Returns:
        X: list of headline strings
        y: list of int labels  (fox=0, nbc=1)
    """
    df = pd.read_csv(path)

    if "url" in df.columns:
        X = df["url"].fillna("").astype(str).apply(_headline_from_url).tolist()
    elif "headline" in df.columns:
        X = df["headline"].fillna("").astype(str).tolist()
    else:
        raise ValueError(f"CSV at '{path}' must have a 'headline' or 'url' column. Found: {list(df.columns)}")

    if "label" in df.columns:
        y = df["label"].astype(int).tolist()
    elif "news_source" in df.columns:
        sources = df["news_source"].fillna("").astype(str).str.lower()
        y = sources.apply(lambda s: 0 if "fox" in s else 1 if "nbc" in s else 0).tolist()
    elif "url" in df.columns:
        urls = df["url"].fillna("").astype(str).str.lower()
        y = urls.apply(lambda u: _label_from_source_text(u) if _label_from_source_text(u) is not None else 0).tolist()
    else:
        y = [0] * len(X)

    return X, y
