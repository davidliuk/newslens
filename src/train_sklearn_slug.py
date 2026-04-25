"""
Train a URL-slug text classifier for leaderboard submissions.

The leaderboard provides URL-only CSVs but rejects raw URLs/domains as features.
This script mirrors preprocess.py by converting each URL path into readable text,
then trains a compact TF-IDF + LinearSVC model saved inside model.pt.

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
from sklearn.linear_model import SGDClassifier

from model import Model
from preprocess import _headline_from_url, _label_from_source_text


def main() -> None:
    df = pd.read_csv("data/raw/url_only_data.csv")
    df["text"] = df["url"].fillna("").astype(str).map(_headline_from_url)
    df["label"] = df["url"].fillna("").astype(str).map(_label_from_source_text)
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
        SGDClassifier(loss="hinge", alpha=3e-4, max_iter=3000, tol=1e-4, random_state=1),
    )

    print(f"Training on {len(X)} URL-derived headlines")
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
