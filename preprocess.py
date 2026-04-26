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
    """
    Conservative fallback for URL-only CSVs.

    The leaderboard should provide actual scraped headlines when available. If a
    CSV only has URLs, we use only the final article slug as headline-like text
    and remove outlet/domain/id artifacts.
    """
    path = unquote(urlparse(str(url)).path)
    segments = [seg for seg in path.split("/") if seg]
    if not segments:
        return ""

    segment = segments[-1].lower()
    segment = re.sub(r"\.(html?|print)$", "", segment)
    segment = re.sub(r"\b(?:rcna|ncna|bbna|n)\d+\b", "", segment)
    segment = re.sub(r"\b\d{4}\b", "", segment)
    segment = re.sub(r"[^a-z0-9]+", " ", segment)

    drop = {"www", "foxnews", "nbcnews", "fox", "nbc", "news", "com"}
    return " ".join(
        tok for tok in segment.split()
        if tok and not tok.isdigit() and tok not in drop
    ).strip()


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Load a news-headline CSV and return (headlines, labels).

    Prefer an actual 'headline' column. Raw URLs, domains, and URL path
    categories are not used as model features.
    """
    df = pd.read_csv(path)

    if "headline" in df.columns:
        X = df["headline"].fillna("").astype(str).tolist()
    elif "url" in df.columns:
        X = df["url"].fillna("").astype(str).apply(_headline_from_url).tolist()
    else:
        raise ValueError(f"CSV at '{path}' must have a 'headline' or 'url' column. Found: {list(df.columns)}")

    if "label" in df.columns:
        y = df["label"].astype(int).tolist()
    elif "news_source" in df.columns:
        sources = df["news_source"].fillna("").astype(str).str.lower()
        y = sources.apply(lambda s: 0 if "fox" in s else 1 if "nbc" in s else 0).tolist()
    elif "url" in df.columns:
        urls = df["url"].fillna("").astype(str)
        y = urls.apply(lambda u: _label_from_source_text(u) if _label_from_source_text(u) is not None else 0).tolist()
    else:
        y = [0] * len(X)

    return X, y
