import pickle
import re
from typing import Any, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters ───────────────────────────────────────────────────────────
PAD_IDX   = 0
UNK_IDX   = 1
EMBED_DIM    = 100  # matches GloVe 6B 100d
NUM_FILTERS  = 128
KERNEL_SIZES = [2, 3, 4, 5]
MAX_LEN      = 40

# Fixed-size buffer for (vocab dict + embedding weights) via pickle.
# Vocab JSON + emb weights (15k × 128 × 4 B) ≈ 8 MB → 12 MB is safe.
_VOCAB_BUF = 12 * 1024 * 1024
_SKLEARN_BUF = 8 * 1024 * 1024


def _tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class Model(nn.Module):
    """
    TextCNN for news headline source classification (fox=0, nbc=1).

    Architecture:
        Embedding → Conv1d (k=2,3,4,5) → ReLU → GlobalMaxPool
        → Concat → Dropout(0.6) → Linear(512, 2)

    state_dict layout (leaderboard-compatible):
        _vocab_bytes  : fixed 12MB uint8 tensor  — pickled {word2idx, emb_weight}
        _vocab_size   : scalar long tensor        — actual byte count in buffer
        convs.*.weight/bias, fc.weight/bias       — standard nn params
        (embedding.weight is stored inside _vocab_bytes, not as a separate key)
    """

    def __init__(self, weights_path: str = None) -> None:
        super().__init__()
        self.word2idx: dict = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}

        # Placeholder embedding — resized in build_vocab / load_state_dict
        self.embedding = nn.Embedding(2, EMBED_DIM, padding_idx=PAD_IDX)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding.weight.data[PAD_IDX].zero_()

        self.convs = nn.ModuleList([
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, kernel_size=k)
            for k in KERNEL_SIZES
        ])
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(NUM_FILTERS * len(KERNEL_SIZES), 2)
        self._fitted = False
        self.sklearn_model = None

        if weights_path and weights_path != "__no_weights__.pth":
            try:
                sd = torch.load(weights_path, map_location="cpu", weights_only=False)
                self.load_state_dict(sd)
            except Exception:
                pass

    # ── Vocab ─────────────────────────────────────────────────────────────────

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        from collections import Counter
        counts = Counter(tok for t in texts for tok in _tokenize(t))
        self.word2idx = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        for word, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if cnt >= min_freq:
                self.word2idx[word] = len(self.word2idx)
        vocab_size = len(self.word2idx)
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding.weight.data[PAD_IDX].zero_()

    def _encode(self, texts: List[str]) -> torch.Tensor:
        seqs = []
        for text in texts:
            toks = _tokenize(text)[:MAX_LEN]
            ids  = [self.word2idx.get(t, UNK_IDX) for t in toks]
            ids += [PAD_IDX] * (MAX_LEN - len(ids))
            seqs.append(ids)
        return torch.tensor(seqs, dtype=torch.long)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)   # (B, E, L)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                  # (B, F, L-k+1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (B, F)
            pooled.append(c)
        out = torch.cat(pooled, dim=1)             # (B, F*4)
        return self.fc(self.dropout(out))          # (B, 2)

    # ── state_dict: vocab+emb in fixed buffer, conv/fc as normal tensors ──────

    def state_dict(self, **kwargs):
        sk_data = pickle.dumps(self.sklearn_model) if self.sklearn_model is not None else b""
        sk_size = len(sk_data)
        assert sk_size <= _SKLEARN_BUF, f"sklearn model too large: {sk_size} B > {_SKLEARN_BUF} B"
        sk_padded = bytearray(sk_data) + bytearray(_SKLEARN_BUF - sk_size)

        data = pickle.dumps({
            "word2idx":   self.word2idx,
            "emb_weight": self.embedding.weight.data.cpu().numpy(),
        })
        size = len(data)
        assert size <= _VOCAB_BUF, f"Vocab+emb too large: {size} B > {_VOCAB_BUF} B"
        padded = bytearray(data) + bytearray(_VOCAB_BUF - size)

        sd = {
            "_sklearn_bytes": torch.frombuffer(sk_padded, dtype=torch.uint8).clone(),
            "_sklearn_size":  torch.tensor(sk_size, dtype=torch.long),
            "_vocab_bytes": torch.frombuffer(padded, dtype=torch.uint8).clone(),
            "_vocab_size":  torch.tensor(size, dtype=torch.long),
        }
        # conv/fc weights — exclude embedding.weight (stored in _vocab_bytes)
        for k, v in super().state_dict(**kwargs).items():
            if k != "embedding.weight":
                sd[k] = v
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        if "_sklearn_bytes" in state_dict and "_sklearn_size" in state_dict:
            size = int(state_dict["_sklearn_size"].item())
            if size > 0:
                raw = state_dict["_sklearn_bytes"][:size].numpy().tobytes()
                self.sklearn_model = pickle.loads(raw)
                self._fitted = True

        if "_vocab_bytes" in state_dict and "_vocab_size" in state_dict:
            size = int(state_dict["_vocab_size"].item())
            raw  = state_dict["_vocab_bytes"][:size].numpy().tobytes()
            obj  = pickle.loads(raw)

            self.word2idx = obj["word2idx"]
            vocab_size    = len(self.word2idx)
            self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
            self.embedding.weight.data = torch.tensor(obj["emb_weight"])

        nn_sd = {k: v for k, v in state_dict.items() if not k.startswith("_vocab") and not k.startswith("_sklearn")}
        if nn_sd:
            super().load_state_dict(nn_sd, strict=False)
            self._fitted = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def eval(self) -> "Model":
        super().eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = [str(t) for t in batch]
        if self.sklearn_model is not None:
            return [int(pred) for pred in self.sklearn_model.predict(texts)]

        if not self._fitted:
            return [0] * len(texts)
        x = self._encode(texts)
        super().eval()
        with torch.no_grad():
            logits = self.forward(x)
        return logits.argmax(dim=1).tolist()


def get_model() -> Model:
    return Model()
