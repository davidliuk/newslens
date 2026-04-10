import io
import pickle
from typing import Any, Iterable, List

import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Model(nn.Module):
    """
    TF-IDF + Logistic Regression baseline for news source classification.

    Wraps a sklearn Pipeline so it is compatible with the leaderboard evaluator:
    - Instantiable with no arguments
    - state_dict() / load_state_dict() encode the fitted pipeline as a uint8 tensor
    - predict(batch) returns a list of int labels (fox=0, nbc=1)
    """

    def __init__(self, weights_path: str = None) -> None:
        super().__init__()
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2),
            )),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        self._fitted = False

        if weights_path and weights_path != "__no_weights__.pth":
            try:
                sd = torch.load(weights_path, map_location="cpu", weights_only=False)
                self.load_state_dict(sd)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # sklearn pipeline persistence via torch-compatible state_dict
    # ------------------------------------------------------------------

    def state_dict(self, **kwargs):
        buf = io.BytesIO()
        pickle.dump(self.pipeline, buf)
        pipeline_bytes = torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)
        return {
            "pipeline_bytes": pipeline_bytes,
            "fitted": torch.tensor(int(self._fitted)),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        if "pipeline_bytes" in state_dict:
            data = state_dict["pipeline_bytes"]
            if isinstance(data, torch.Tensor):
                data = data.numpy().tobytes()
            self.pipeline = pickle.loads(data)
        if "fitted" in state_dict:
            val = state_dict["fitted"]
            self._fitted = bool(val.item() if isinstance(val, torch.Tensor) else val)

    # ------------------------------------------------------------------
    # Training (called from src/train.py, not by the evaluator)
    # ------------------------------------------------------------------

    def fit(self, X: List[str], y: List[int]) -> "Model":
        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def eval(self) -> "Model":
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = [str(t) for t in batch]
        if not self._fitted:
            return [0] * len(texts)
        return self.pipeline.predict(texts).tolist()


def get_model() -> Model:
    return Model()
