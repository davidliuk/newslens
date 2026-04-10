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
    #
    # The leaderboard evaluator shape-checks tensors before passing them to
    # load_state_dict, so pipeline_bytes must always be the same size.
    # We use a fixed 5MB padded buffer and store the real size separately.
    # ------------------------------------------------------------------

    _BUF_SIZE = 5 * 1024 * 1024  # 5MB — well above any realistic sklearn pipeline

    def state_dict(self, **kwargs):
        buf = io.BytesIO()
        pickle.dump(self.pipeline, buf)
        data = buf.getvalue()
        size = len(data)
        assert size <= self._BUF_SIZE, f"Pipeline too large: {size} bytes > {self._BUF_SIZE}"
        padded = bytearray(data) + bytearray(self._BUF_SIZE - size)
        return {
            "pipeline_bytes": torch.frombuffer(padded, dtype=torch.uint8).clone(),
            "pipeline_size": torch.tensor(size, dtype=torch.long),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        if "pipeline_bytes" in state_dict and "pipeline_size" in state_dict:
            size = int(state_dict["pipeline_size"].item())
            data = state_dict["pipeline_bytes"][:size].numpy().tobytes()
            self.pipeline = pickle.loads(data)
            self._fitted = True

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
