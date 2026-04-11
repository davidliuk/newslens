# NewsLens

**CIS 4190/5190 Applied Machine Learning — Spring 2026 Final Project**  
Project B: News Source Classification (Fox News vs. NBC News)  
**Group ID:** 57

## Team

| Name | Role | GitHub |
|------|------|--------|
| Steven Gavin Sears | Data Engineer | [@Gavin-Sears](https://github.com/Gavin-Sears) |
| Dawei Liu | Model Trainer | [@davidliuk](https://github.com/davidliuk) |
| Johnny Ding | Analysis & Report | [@JohnnySIST](https://github.com/JohnnySIST) |

## Task

Binary text classification: given a news headline, predict whether it came from **Fox News** (0) or **NBC News** (1).

## Model

**TextCNN** with GloVe 6B 100d pre-trained embeddings (Kim 2014).

```
Headline text
    │
    ▼
Tokenize  (lowercase, strip punctuation, split)
    │
    ▼
Embedding  [vocab=3858, dim=100, init=GloVe 6B 100d]
    │
    ├─ Conv1d(k=2) → ReLU → GlobalMaxPool  →  128-d
    ├─ Conv1d(k=3) → ReLU → GlobalMaxPool  →  128-d
    ├─ Conv1d(k=4) → ReLU → GlobalMaxPool  →  128-d
    └─ Conv1d(k=5) → ReLU → GlobalMaxPool  →  128-d
                                                │
                                           Concat  →  512-d
                                                │
                                          Dropout(0.6)
                                                │
                                         Linear(512→2)
                                                │
                                        fox=0  /  nbc=1
```

Each Conv1d branch captures n-gram patterns of a different width (bigrams through 5-grams).
GlobalMaxPool picks the strongest signal regardless of position in the headline.
Embeddings are initialised from GloVe and fine-tuned end-to-end.

| Property | Value |
|----------|-------|
| Vocab size | 3,858 (min\_freq=2) |
| Embedding | GloVe 6B 100d, fine-tuned |
| GloVe coverage | ~82% of vocab |
| Conv filters | 128 per kernel size |
| Kernel sizes | 2, 3, 4, 5 |
| Parameters | ~330K |
| Optimizer | Adam (lr=5e-4, wd=1e-3) |
| Regularization | Dropout 0.6, grad clip 1.0, early stopping (patience=10) |

### Experiment Results

| Model | Val Accuracy |
|-------|-------------|
| Course baseline: TF-IDF (100 features) + LogReg | 66.49% |
| TF-IDF (5k features, bigrams) + LogReg | 79.1% |
| TextCNN — random init (EMBED=64) | 81.3% |
| BiLSTM — single layer, CPU only† | 78.8% |
| **TextCNN + GloVe 6B 100d (current)** | **82.8%** |

†PyTorch LSTM on MPS (Apple Silicon) has a known bug; must run on CPU.

## Dataset

Hosted on Hugging Face: [SGavin/CIS5190_NewsSource](https://huggingface.co/datasets/SGavin/CIS5190_NewsSource)

| Property | Value |
|----------|-------|
| Total rows | 3,801 |
| Split | `train` only |
| Labels | `fox` / `nbc` |
| Columns | `news_source`, `headline`, `url` |
| Headline length | 22–238 characters |

```python
from datasets import load_dataset

ds = load_dataset("SGavin/CIS5190_NewsSource")
df = ds["train"].to_pandas()
df["label"] = df["news_source"].map({"fox": 0, "nbc": 1})
```

## Repository Structure

```
newslens/
├── data/
│   ├── raw/              # url_only_data.csv (3,815 article URLs)
│   ├── processed/        # train/val CSVs — git-ignored
│   └── glove/            # GloVe vectors — git-ignored, auto-downloaded
├── src/
│   ├── glove.py          # GloVe download + embedding matrix loader
│   ├── train_cnn.py      # TextCNN training (primary)
│   ├── train_lstm.py     # BiLSTM experiment (comparison)
│   └── train.py          # TF-IDF baseline training
├── models/               # saved checkpoints — git-ignored
├── notebooks/            # EDA and analysis notebooks
├── figures/              # report plots
├── model.py              # leaderboard: TextCNN Model class
├── preprocess.py         # leaderboard: prepare_data()
├── eval_project_b.py     # course-provided local evaluator
└── proposal.md
```

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync

# Train TextCNN (GloVe ~822MB downloaded automatically on first run)
uv run python src/train_cnn.py

# Local evaluation
uv run python eval_project_b.py \
  --model model.py --preprocess preprocess.py \
  --csv data/processed/val.csv --weights models/text_cnn.pt
```

> GloVe vectors are only needed during training. The fine-tuned embeddings are
> saved inside `models/text_cnn.pt` and are not required at inference time.

## Hardware

Runs on **Apple Silicon (M4)** via PyTorch MPS backend — auto-detected at training time.  
Training completes in under 30 seconds (TextCNN, ~30 epochs on MPS).

## Leaderboard Submission

Leaderboard: [cis4190/NewsHeadlineClassifier](https://huggingface.co/spaces/cis4190/NewsHeadlineClassifier)

Upload the following in the **Student Submissions** tab:

| Field | Value |
|-------|-------|
| Group ID | `57` |
| Alias | `newslens` |
| State Dict | `models/text_cnn.pt` |
| model.py | `model.py` |
| preprocess.py | `preprocess.py` |

Evaluation runs on `url_val` and `url_val16k` datasets and reports accuracy + inference time.  
Check submission status under the **Submission Status** tab using Group ID `57`.

## Submission Deadline

May 6, 2026
