"""
Train the baseline TF-IDF + Logistic Regression model on the HuggingFace dataset
and save weights to models/tfidf_logreg.pt.

Usage:
    uv run python src/train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from datasets import load_dataset

# ── Device detection (MPS on Apple Silicon, CUDA on Linux, else CPU) ──────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from model import Model

# ── Load dataset from HuggingFace ────────────────────────────────────────────
print("Loading dataset SGavin/CIS5190_NewsSource ...")
ds = load_dataset("SGavin/CIS5190_NewsSource")
df = ds["train"].to_pandas()

df["label"] = df["news_source"].map({"fox": 0, "nbc": 1})
df = df.dropna(subset=["headline", "label"])
df["label"] = df["label"].astype(int)

print(f"Total examples: {len(df)}")
print(df["news_source"].value_counts().to_string())

# ── Split ─────────────────────────────────────────────────────────────────────
X = df["headline"].tolist()
y = df["label"].tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}")

# ── Save val split for local eval ─────────────────────────────────────────────
import pandas as pd
os.makedirs("data/processed", exist_ok=True)
val_df = pd.DataFrame({"headline": X_val, "label": y_val})
val_df.to_csv("data/processed/val.csv", index=False)
print("Saved data/processed/val.csv")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining TF-IDF + Logistic Regression ...")
model = Model()
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_val)
acc = sum(p == t for p, t in zip(preds, y_val)) / len(y_val)
print(f"\nValidation accuracy: {acc:.4f}")
print(classification_report(y_val, preds, target_names=["fox", "nbc"]))

# ── Save weights ──────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
out_path = "models/tfidf_logreg.pt"
torch.save(model.state_dict(), out_path)
print(f"Saved model weights → {out_path}")
print(f"\nTo evaluate locally:")
print(f"  python eval_project_b.py --model model.py --preprocess preprocess.py \\")
print(f"    --csv data/processed/val.csv --weights {out_path}")
