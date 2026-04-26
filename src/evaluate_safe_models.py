"""
Evaluate safe headline-only models.

This script uses URLs only to recover the scraped headline and to build
validation splits. Model features are always headline text, not URL paths,
domains, or publisher-specific URL structure.

Usage:
    uv run python src/evaluate_safe_models.py
"""
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from preprocess import _headline_from_url, _label_from_source_text


@dataclass
class EvalResult:
    split: str
    model: str
    accuracy: float
    f1_macro: float


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/raw/url_only_data.csv")
    df["headline"] = df["url"].fillna("").astype(str).map(_headline_from_url)
    df["label"] = df["url"].fillna("").astype(str).map(_label_from_source_text)
    df = df.dropna(subset=["headline", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["topic"] = df["url"].fillna("").astype(str).map(_topic_from_url)
    return df


def _topic_from_url(url: str) -> str:
    parts = [part for part in urlparse(str(url)).path.split("/") if part]
    return parts[0].lower() if parts else "unknown"


def _tfidf_union(
    word_ngram=(1, 3),
    char_ngram=(3, 6),
    word_max_features: int = 50_000,
    char_max_features: int = 80_000,
) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=word_ngram,
                    min_df=1,
                    max_features=word_max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=char_ngram,
                    min_df=1,
                    max_features=char_max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def make_models() -> Dict[str, Callable[[], Pipeline]]:
    return {
        "word_sgd": lambda: Pipeline(
            [
                (
                    "features",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 3),
                        min_df=1,
                        max_features=70_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        alpha=3e-4,
                        max_iter=3000,
                        tol=1e-4,
                        random_state=1,
                    ),
                ),
            ]
        ),
        "char_sgd": lambda: Pipeline(
            [
                (
                    "features",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 6),
                        min_df=1,
                        max_features=120_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        alpha=3e-4,
                        max_iter=3000,
                        tol=1e-4,
                        random_state=2,
                    ),
                ),
            ]
        ),
        "word_char_sgd": lambda: Pipeline(
            [
                ("features", _tfidf_union()),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        alpha=3e-4,
                        max_iter=3000,
                        tol=1e-4,
                        random_state=3,
                    ),
                ),
            ]
        ),
        "word_char_svc": lambda: Pipeline(
            [("features", _tfidf_union()), ("clf", LinearSVC(C=3.0, max_iter=5000))]
        ),
        "word_char_ridge": lambda: Pipeline(
            [("features", _tfidf_union()), ("clf", RidgeClassifier(alpha=0.1))]
        ),
        "vote_linear": lambda: VotingClassifier(
            estimators=[
                ("sgd", make_models()["word_char_sgd"]()),
                ("svc", make_models()["word_char_svc"]()),
                ("ridge", make_models()["word_char_ridge"]()),
                (
                    "lr",
                    Pipeline(
                        [
                            ("features", _tfidf_union()),
                            (
                                "clf",
                                LogisticRegression(
                                    C=12,
                                    max_iter=3000,
                                    solver="liblinear",
                                ),
                            ),
                        ]
                    ),
                ),
            ],
            voting="hard",
        ),
    }


def compact_stratify_keys(labels: pd.Series, topics: pd.Series, min_count: int = 2) -> pd.Series:
    keys = labels.astype(str) + "::" + topics.astype(str)
    counts = keys.value_counts()
    return keys.where(keys.map(counts) >= min_count, labels.astype(str) + "::other")


def evaluate_split(name: str, train_idx, val_idx, df: pd.DataFrame) -> List[EvalResult]:
    X_train = df.iloc[train_idx]["headline"].tolist()
    y_train = df.iloc[train_idx]["label"].tolist()
    X_val = df.iloc[val_idx]["headline"].tolist()
    y_val = df.iloc[val_idx]["label"].tolist()

    results: List[EvalResult] = []
    for model_name, factory in make_models().items():
        model = factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        results.append(
            EvalResult(
                split=name,
                model=model_name,
                accuracy=accuracy_score(y_val, preds),
                f1_macro=f1_score(y_val, preds, average="macro"),
            )
        )
    return results


def main() -> None:
    df = load_data()
    print(f"Rows: {len(df)}")
    print("Label counts:")
    print(df["label"].value_counts().sort_index().to_string())
    print("Top topics:")
    print(df["topic"].value_counts().head(12).to_string())

    all_results: List[EvalResult] = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(cv.split(df["headline"], df["label"]), start=1):
        all_results.extend(evaluate_split(f"random_cv_{fold}", train_idx, val_idx, df))

    topic_keys = compact_stratify_keys(df["label"], df["topic"])
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=topic_keys,
    )
    all_results.extend(evaluate_split("topic_balanced", train_idx, val_idx, df))

    out = pd.DataFrame([r.__dict__ for r in all_results])
    summary = (
        out.groupby("model")
        .agg(
            random_cv_acc=("accuracy", lambda s: s[out.loc[s.index, "split"].str.startswith("random_cv")].mean()),
            random_cv_f1=("f1_macro", lambda s: s[out.loc[s.index, "split"].str.startswith("random_cv")].mean()),
            topic_acc=("accuracy", lambda s: s[out.loc[s.index, "split"].eq("topic_balanced")].mean()),
            topic_f1=("f1_macro", lambda s: s[out.loc[s.index, "split"].eq("topic_balanced")].mean()),
        )
        .sort_values(["topic_acc", "random_cv_acc"], ascending=False)
    )

    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/safe_model_eval_raw.csv", index=False)
    summary.to_csv("reports/safe_model_eval_summary.csv")
    print("\nSummary:")
    print(summary.round(4).to_string())
    print("\nSaved reports/safe_model_eval_raw.csv and reports/safe_model_eval_summary.csv")


if __name__ == "__main__":
    main()
