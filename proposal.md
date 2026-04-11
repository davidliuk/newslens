# CIS 4190/5190 Final Project Proposal
## Project B: News Source Classification

**Course:** CIS 4190/5190 Applied Machine Learning — Spring 2026  
**Submission Deadline:** May 6, 2026  
**Team Size:** 3 members  
**Group ID:** 57  
**Leaderboard:** [cis4190/NewsHeadlineClassifier](https://huggingface.co/spaces/cis4190/NewsHeadlineClassifier)

---

## 1. Problem Statement

We aim to build a binary text classifier that distinguishes news headlines from **Fox News** (label 0) vs. **NBC News** (label 1) based on headline text alone. The course provides 3,815 article URLs (2,010 from Fox News, 1,805 from NBC News) as a starting point for web scraping. We must surpass the provided baseline accuracy of **66.49%** (TF-IDF + Logistic Regression).

The evaluation set consists of unseen headlines scraped from both outlets after the submission deadline, so our model must generalize beyond the provided URLs.

Dataset: [SGavin/CIS5190_NewsSource](https://huggingface.co/datasets/SGavin/CIS5190_NewsSource) — 3,801 scraped headlines (fox/nbc labels).

---

## 2. Implementation Plan

### 2.1 Data Collection

- Web-scraped headlines from the 3,815 provided URLs using `requests` + `BeautifulSoup`.
- Handled failed requests (non-200 status codes, changed page structure) by logging and skipping.
- Extracted `<h1>` headline tags; stored as CSV with columns `headline`, `news_source`, `url`.
- Published dataset on HuggingFace: [SGavin/CIS5190_NewsSource](https://huggingface.co/datasets/SGavin/CIS5190_NewsSource).

### 2.2 Data Cleaning & Preprocessing

| Step | Description |
|------|-------------|
| Deduplication | Remove exact-duplicate headlines |
| Missing data | Drop rows where headline is null or empty |
| Normalization | Lowercase; strip punctuation and non-alphanumeric characters |
| Train/Val split | 80/20 stratified split (`random_state=42`) |
| Class balance | Fox: 2,000 / NBC: 1,801 — slight imbalance, handled via stratified split |
| Tokenization | Word-level; min\_freq=2 vocabulary (3,858 tokens) |

### 2.3 Model Design

Incremental model progression — each stage builds on findings from the previous:

#### Stage 1 — Course Baseline ✅
- TF-IDF (max 100 features) + Logistic Regression
- Result: **66.49%** accuracy

#### Stage 2 — Improved Classical ML ✅
- TF-IDF (5,000 features, bigrams, sublinear_tf) + Logistic Regression
- Result: **79.1%** accuracy (+12.6pp)

#### Stage 3 — TextCNN from Scratch ✅
- Embedding(3858, 64) → Conv1d(k=2,3,4,5, 128 filters) → GlobalMaxPool → Dropout(0.6) → Linear
- Adam (lr=5e-4, wd=1e-3), early stopping (patience=10)
- Result: **81.3%** accuracy

#### Stage 4 — TextCNN + GloVe Embeddings ✅ *(current best)*
- Same architecture with EMBED\_DIM=100, embeddings initialised from GloVe 6B 100d
- GloVe loaded from flat text file using only Python built-ins + numpy (no extra library)
- Fine-tuned end-to-end; 82% of vocab words found in GloVe
- Result: **82.8%** accuracy

#### Stage 5 — Ensemble (planned)
- Average softmax probabilities from TextCNN + TF-IDF/LogReg
- Expected: +1–2% additional improvement

#### Comparison — BiLSTM ✅
- Single-layer BiLSTM (hidden=64, bidirectional), max pool over hidden states
- Result: **78.8%** — worse than TextCNN for short headlines
- Note: PyTorch LSTM has a known MPS bug on Apple Silicon; must run on CPU

### 2.4 Evaluation Protocol

- **Primary metric:** Classification accuracy on the course leaderboard
- **Reported metrics:** Accuracy, Precision, Recall, F1-score (macro and per-class)
- **Internal evaluation:** Stratified 80/20 val split; early stopping monitors val accuracy
- **Leaderboard:** Submit at least one entry; iterate to improve ranking

---

## 3. Exploratory Component

**Analysis** — *What linguistic features most distinguish Fox News from NBC News headlines?*

- Ablation study: compare baseline → improved TF-IDF → TextCNN (scratch) → TextCNN + GloVe
- Architecture comparison: TextCNN vs BiLSTM — why CNN outperforms on short text
- Top discriminative n-grams per class (TF-IDF feature weights)
- Error analysis: manual inspection of misclassified headlines, per-class confusion patterns
- Headline length distribution and its correlation with classification difficulty

---

## 4. Team Division of Labor

| Role | Member | GitHub | Responsibilities |
|------|--------|--------|-----------------|
| **Data Engineer** | Steven Gavin Sears | [@Gavin-Sears](https://github.com/Gavin-Sears) | Web scraping, data cleaning pipeline, HuggingFace dataset publishing, data statistics |
| **Model Trainer** | Dawei Liu | [@davidliuk](https://github.com/davidliuk) | All model implementations, hyperparameter tuning, GloVe integration, leaderboard submissions, experiment logs |
| **Analysis & Report** | Johnny Ding | [@JohnnySIST](https://github.com/JohnnySIST) | Error analysis, ablation study, figures (model comparison charts, confusion matrix), final report writing |

> Gavin hands off the cleaned dataset → Dawei trains and logs experiments → Johnny analyzes results and writes the report.

---

## 5. Milestones

| Date | Milestone | Status | Owner |
|------|-----------|--------|-------|
| **Apr 10** | Gradescope team formation submitted | ✅ | All |
| **Apr 10** | Dataset scraped & published on HuggingFace | ✅ | Gavin |
| **Apr 10** | Baseline reproduced (66.49%) | ✅ | Dawei |
| **Apr 10** | TextCNN + GloVe implemented (82.8%) | ✅ | Dawei |
| **Apr 25** | First leaderboard submission | ⬜ | Dawei |
| **Apr 27** | Ensemble model or further improvements | ⬜ | Dawei |
| **Apr 28** | Error analysis, figures drafted | ⬜ | Johnny |
| **May 1** | Report first draft | ⬜ | Johnny |
| **May 4** | Report finalized; all artifacts packaged | ⬜ | All |
| **May 6** | Final submission on Gradescope | ⬜ | All |

---

## 6. Deliverables

- [x] **Dataset:** [SGavin/CIS5190_NewsSource](https://huggingface.co/datasets/SGavin/CIS5190_NewsSource) on HuggingFace
- [ ] **Best model:** `models/text_cnn.pt` (TextCNN + GloVe, 82.8% val accuracy)
- [ ] **Project report:** 5-page report covering data, model, evaluation, and analysis
- [ ] **Leaderboard entry:** At least one submission before May 6

---

## 7. Tools & Infrastructure

| Component | Tool |
|-----------|------|
| Environment | `uv` (Python 3.12) |
| Web scraping | `requests`, `BeautifulSoup4` |
| Data loading | `datasets` (HuggingFace), `pandas` |
| Classical ML | `scikit-learn` |
| Deep learning | `torch` (MPS on Apple Silicon M4) |
| Word vectors | GloVe 6B 100d (Stanford, loaded via `numpy`) |
| Experiment tracking | Manual logs in `src/train_*.py` output |
| Version control | Git |
| Analysis & figures | `matplotlib`, `seaborn`, `shap` |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Scraping failures (paywalls, 404s) | Logged and skipped; 3,801/3,815 URLs successfully scraped |
| Class imbalance | Stratified splits; per-class F1 reported alongside accuracy |
| Overfitting (small dataset) | Dropout 0.6, weight decay 1e-3, early stopping, min\_freq vocab filtering |
| MPS bugs (LSTM) | LSTM experiment run on CPU; primary model (CNN) works on MPS |
| Leaderboard backend missing `transformers` | Using only `torch` + `scikit-learn`; no transformer dependency |
| Deadline risk | All roles independently executable; model already at 82.8% |
