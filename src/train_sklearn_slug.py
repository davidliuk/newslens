"""
Train a headline-only classifier for leaderboard submissions.

Train on the locally exported Hugging Face dataset:
data/raw/news_source_headlines.csv with columns news_source, headline, url.
Only the actual scraped headline is used as a model feature; urls are kept in
the dataset for provenance and splitting, not for training.

Usage:
    uv run python src/train_sklearn_slug.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from model import Model


def main() -> None:
    df = pd.read_csv("data/raw/news_source_headlines.csv")
    df = df.dropna(subset=["headline", "news_source"]).copy()
    df["text"] = df["headline"].fillna("").astype(str)
    df["label"] = df["news_source"].astype(str).str.lower().map({"fox": 0, "nbc": 1})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    X = df["text"].tolist()
    y = df["label"].tolist()

    pipeline = make_pipeline(
        FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 3),
                        min_df=1,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 6),
                        min_df=1,
                        max_features=80_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        ),
        LinearSVC(C=3.0, max_iter=5000),
    )

    print(f"Training on {len(X)} scraped headlines")
    print(df["label"].value_counts().sort_index().to_string())
    pipeline.fit(X, y)

    model = Model()
    model.sklearn_model = pipeline
    model._fitted = True

    os.makedirs("models", exist_ok=True)
    out_path = "models/model.pt"
    torch.save(model.state_dict(), out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
