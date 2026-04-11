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

**TextCNN** — a convolutional neural network for short text classification (Kim 2014).

```
Headline text
    │
    ▼
Tokenize (lowercase, strip punctuation)
    │
    ▼
Embedding  [vocab=3858, dim=64]
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

Each Conv1d branch captures n-gram patterns of a different width (bigrams through 5-grams). GlobalMaxPool picks the strongest signal regardless of position. The four branches are concatenated before classification.

| Property | Value |
|----------|-------|
| Vocab size | 3,858 (min_freq=2) |
| Embedding dim | 64 |
| Conv filters | 128 per kernel |
| Kernel sizes | 2, 3, 4, 5 |
| Parameters | ~280K |
| Optimizer | Adam (lr=5e-4, wd=1e-3) |
| Regularization | Dropout 0.6, grad clip 1.0, early stopping |

### Results

| Model | Val Accuracy |
|-------|-------------|
| Baseline: TF-IDF (100 features) + LogReg | 66.49% |
| Improved: TF-IDF (5k, bigrams) + LogReg | 79.1% |
| **TextCNN (current)** | **81.3%** |

## Dataset

Hosted on Hugging Face: [SGavin/CIS5190_NewsSource](https://huggingface.co/datasets/SGavin/CIS5190_NewsSource)

| Property | Value |
|----------|-------|
| Total rows | 3,801 |
| Split | `train` only |
| Labels | `fox` / `nbc` |
| Columns | `news_source`, `headline`, `url` |
| Headline length | 22–238 characters |

### Load with `datasets`

```python
from datasets import load_dataset

ds = load_dataset("SGavin/CIS5190_NewsSource")
df = ds["train"].to_pandas()
# df columns: news_source, headline, url
```

### Load with `pandas`

```python
import pandas as pd

df = pd.read_csv(
    "hf://datasets/SGavin/CIS5190_NewsSource/data/train-00000-of-00001.parquet"
)
# or via the Hugging Face hub:
# df = pd.read_parquet("hf://datasets/SGavin/CIS5190_NewsSource/data/train-00000-of-00001.parquet")
```

### Label encoding

The leaderboard uses numeric labels: `fox` → `0`, `nbc` → `1`.

```python
df["label"] = df["news_source"].map({"fox": 0, "nbc": 1})
```

## Repository Structure

```
newslens/
├── data/
│   ├── raw/              # url_only_data.csv (3,815 article URLs)
│   └── processed/        # cleaned train/val/test CSVs (git-ignored)
├── notebooks/            # EDA, scraping, and experiment notebooks
├── src/                  # reusable utility modules
├── models/               # saved checkpoints (git-ignored)
├── figures/              # plots and charts for the report
├── model.py              # leaderboard submission: Model class
├── preprocess.py         # leaderboard submission: prepare_data()
├── eval_project_b.py     # course-provided evaluation script
├── proposal.md           # project proposal
└── requirements.txt
```

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync

# Train TextCNN
uv run python src/train_cnn.py

# Local evaluation
uv run python eval_project_b.py \
  --model model.py --preprocess preprocess.py \
  --csv data/processed/val.csv --weights models/text_cnn.pt
```

## Hardware

Runs on **Apple Silicon (M4)** via PyTorch MPS backend — auto-detected at training time.

## Leaderboard Submission

Leaderboard: [cis4190/NewsHeadlineClassifier](https://huggingface.co/spaces/cis4190/NewsHeadlineClassifier)

Upload the following three files in the **Student Submissions** tab:

| Field | Value |
|-------|-------|
| Group ID | `57` |
| Alias | `newslens` (or any team nickname) |
| State Dict | `models/text_cnn.pt` |
| model.py | `model.py` |
| preprocess.py | `preprocess.py` |

Evaluation runs on `url_val` and `url_val16k` datasets and reports accuracy + inference time.  
Check submission status and failed runs under the **Submission Status** tab using Group ID `57`.

## Submission Deadline

May 6, 2026
