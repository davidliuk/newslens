"""
Train TextCNN on the HuggingFace dataset.

Usage:
    uv run python src/train_cnn.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

from model import Model

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 80
LR           = 5e-4
WEIGHT_DECAY = 1e-3
PATIENCE     = 10

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading dataset SGavin/CIS5190_NewsSource ...")
ds = load_dataset("SGavin/CIS5190_NewsSource")
df = ds["train"].to_pandas()
df["label"] = df["news_source"].map({"fox": 0, "nbc": 1}).astype(int)
df = df.dropna(subset=["headline", "label"])

X = df["headline"].tolist()
y = df["label"].tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}")

os.makedirs("data/processed", exist_ok=True)
pd.DataFrame({"headline": X_val, "label": y_val}).to_csv("data/processed/val.csv", index=False)

# ── Build model & vocab ───────────────────────────────────────────────────────
model = Model()
model.build_vocab(X_train, min_freq=2)
print(f"Vocab size: {len(model.word2idx)}")

# ── Encode ────────────────────────────────────────────────────────────────────
X_train_t = model._encode(X_train)
X_val_t   = model._encode(X_val)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE)

# ── Training ──────────────────────────────────────────────────────────────────
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss()

best_acc        = 0.0
patience_count  = 0
# Store only trainable params during loop (avoids vocab pickle overhead per epoch)
best_params: dict = {}

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
    val_acc = correct / len(X_val)
    avg_loss = total_loss / len(train_loader)
    scheduler.step(1 - val_acc)

    print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.4f}"
          + (" ✓ best" if val_acc > best_acc else ""))

    if val_acc > best_acc:
        best_acc = val_acc
        best_params = {k: v.cpu().clone() for k, v in model.named_parameters()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

# ── Restore best weights ──────────────────────────────────────────────────────
model = model.cpu()
for name, param in model.named_parameters():
    if name in best_params:
        param.data.copy_(best_params[name])
model._fitted = True

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_val)
print(f"\nBest val accuracy: {best_acc:.4f}")
print(classification_report(y_val, preds, target_names=["fox", "nbc"]))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
out_path = "models/text_cnn.pt"
torch.save(model.state_dict(), out_path)
print(f"Saved → {out_path}")
print(f"\nTo evaluate locally:")
print(f"  uv run python eval_project_b.py --model model.py --preprocess preprocess.py \\")
print(f"    --csv data/processed/val.csv --weights {out_path}")
