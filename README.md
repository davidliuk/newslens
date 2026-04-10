# NewsLens

**CIS 4190/5190 Applied Machine Learning — Spring 2026 Final Project**  
Project B: News Source Classification (Fox News vs. NBC News)

## Team

| Name | Role | GitHub |
|------|------|--------|
| Steven Gavin Sears | Data Engineer | [@Gavin-Sears](https://github.com/Gavin-Sears) |
| Dawei Liu | Model Trainer | [@davidliuk](https://github.com/davidliuk) |
| Johnny Ding | Analysis & Report | [@JohnnySIST](https://github.com/JohnnySIST) |

## Task

Binary text classification: given a news headline, predict whether it came from **Fox News** (0) or **NBC News** (1).

Baseline: TF-IDF + Logistic Regression — 66.49% accuracy. We aim to exceed this with fine-tuned transformer models.

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

# Train baseline model
uv run python src/train.py

# Local evaluation
uv run python eval_project_b.py \
  --model model.py --preprocess preprocess.py \
  --csv data/processed/val.csv --weights models/tfidf_logreg.pt
```

## Hardware

Runs on **Apple Silicon (M4)** via PyTorch MPS backend.  
The TF-IDF baseline trains on CPU in seconds. Future transformer fine-tuning will use MPS automatically.

## Submission Deadline

May 6, 2026
