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
pip install -r requirements.txt
```

## Submission Deadline

May 6, 2026
