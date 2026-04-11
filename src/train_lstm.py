"""
BiLSTM experiment — compare against TextCNN on the same data/vocab split.

Usage:
    uv run python src/train_lstm.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

from model import Model as TextCNN, _tokenize, PAD_IDX, UNK_IDX, EMBED_DIM, MAX_LEN

# ── Device ────────────────────────────────────────────────────────────────────
# MPS has known bugs with LSTM — force CPU for this experiment
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_DIM   = 64
NUM_LAYERS   = 1
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 80
LR           = 2e-3
WEIGHT_DECAY = 1e-3
PATIENCE     = 10


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding.weight.data[PAD_IDX].zero_()

        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 2)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                          # (B, L, E)
        out, _ = self.lstm(emb)                          # (B, L, 2H)
        # Max pool over all timesteps — captures the strongest signal
        pooled = out.max(dim=1).values                   # (B, 2H)
        return self.fc(self.dropout(pooled))             # (B, 2)


# ── Data (same split as CNN experiment) ──────────────────────────────────────
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

# Build vocab via TextCNN helper (same settings as CNN experiment)
_cnn = TextCNN()
_cnn.build_vocab(X_train, min_freq=2)
word2idx  = _cnn.word2idx
vocab_size = len(word2idx)
print(f"Vocab size: {vocab_size}")

def encode(texts):
    seqs = []
    for text in texts:
        toks = _tokenize(text)[:MAX_LEN]
        ids  = [word2idx.get(t, UNK_IDX) for t in toks]
        ids += [PAD_IDX] * (MAX_LEN - len(ids))
        seqs.append(ids)
    return torch.tensor(seqs, dtype=torch.long)

X_train_t = encode(X_train)
X_val_t   = encode(X_val)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE)

# ── Train ─────────────────────────────────────────────────────────────────────
model = BiLSTMClassifier(vocab_size).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss()

best_acc       = 0.0
patience_count = 0
best_params    = {}

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
            correct += (model(xb.to(DEVICE)).argmax(1) == yb.to(DEVICE)).sum().item()
    val_acc  = correct / len(X_val)
    avg_loss = total_loss / len(train_loader)
    scheduler.step(1 - val_acc)

    print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.4f}"
          + (" ✓ best" if val_acc > best_acc else ""))

    if val_acc > best_acc:
        best_acc       = val_acc
        best_params    = {k: v.cpu().clone() for k, v in model.named_parameters()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

# ── Evaluate ──────────────────────────────────────────────────────────────────
model = model.cpu()
for name, param in model.named_parameters():
    if name in best_params:
        param.data.copy_(best_params[name])
model.eval()

with torch.no_grad():
    preds = model(X_val_t).argmax(1).tolist()

print(f"\nBest val accuracy: {best_acc:.4f}")
print(classification_report(y_val, preds, target_names=["fox", "nbc"]))
print(f"\n── Comparison ───────────────────────────────")
print(f"  Baseline TF-IDF + LogReg : 66.49%")
print(f"  Improved TF-IDF + LogReg : 79.10%")
print(f"  TextCNN  (current model) : 81.34%")
print(f"  BiLSTM   (this run)      : {best_acc*100:.2f}%")
