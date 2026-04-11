"""
GloVe 6B 100d download and loading utility.

GloVe is only needed at training time — fine-tuned embeddings are saved
inside models/text_cnn.pt and do NOT require GloVe at inference/leaderboard.
"""
import os
import zipfile
import urllib.request
from typing import Dict

import numpy as np

GLOVE_URL  = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_DIR  = "data/glove"
GLOVE_FILE = "glove.6B.100d.txt"
GLOVE_DIM  = 100


def _reporthook(block_num, block_size, total_size):
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded / total_size * 100
    filled = int(pct / 2)
    bar = "█" * filled + "░" * (50 - filled)
    print(f"\r  [{bar}] {pct:5.1f}%  {downloaded/1e6:.0f}/{total_size/1e6:.0f} MB",
          end="", flush=True)


def ensure_glove() -> str:
    """
    Download GloVe 6B zip (~822 MB) if not already present, extract the 100d
    file, and delete the zip to save disk space.

    Returns the path to glove.6B.100d.txt.
    """
    os.makedirs(GLOVE_DIR, exist_ok=True)
    txt_path = os.path.join(GLOVE_DIR, GLOVE_FILE)

    if os.path.exists(txt_path):
        print(f"GloVe already present: {txt_path}")
        return txt_path

    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print("Downloading GloVe 6B from Stanford (~822 MB) ...")
        urllib.request.urlretrieve(GLOVE_URL, zip_path, reporthook=_reporthook)
        print()

    print(f"Extracting {GLOVE_FILE} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(GLOVE_FILE, GLOVE_DIR)

    os.remove(zip_path)
    print(f"Saved → {txt_path}  (zip deleted)")
    return txt_path


def load_glove(word2idx: Dict[str, int],
               dim: int = GLOVE_DIM,
               path: str = None) -> np.ndarray:
    """
    Build an embedding matrix from GloVe vectors.

    - Words found in GloVe  →  use pre-trained vector
    - OOV words             →  random N(0, 0.01) init (same as scratch)
    - PAD (index 0)         →  all zeros

    Returns float32 ndarray of shape (vocab_size, dim).
    """
    if path is None:
        path = ensure_glove()

    vocab_size = len(word2idx)
    matrix = np.random.normal(scale=0.01, size=(vocab_size, dim)).astype(np.float32)
    matrix[0] = 0.0  # PAD token stays zero

    hits = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if word in word2idx:
                matrix[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                hits += 1

    pct = hits / vocab_size * 100
    print(f"GloVe coverage: {hits}/{vocab_size} vocab words ({pct:.1f}% hit rate)")
    return matrix
