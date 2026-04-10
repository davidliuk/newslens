# CIS 4190/5190 Final Project Proposal
## Project B: News Source Classification

**Course:** CIS 4190/5190 Applied Machine Learning — Spring 2026  
**Submission Deadline:** May 6, 2026  
**Team Size:** 3 members

---

## 1. Problem Statement

We aim to build a binary text classifier that distinguishes news headlines from **Fox News** (label 0) vs. **NBC News** (label 1) based on headline text alone. The course provides 3,815 article URLs (2,010 from Fox News, 1,805 from NBC News) as a starting point for web scraping. We must surpass the provided baseline accuracy of **66.49%** (TF-IDF + Logistic Regression).

The evaluation set consists of unseen headlines scraped from both outlets after the submission deadline, so our model must generalize beyond the provided URLs.

---

## 2. Implementation Plan

### 2.1 Data Collection

- Use `requests` + `BeautifulSoup` to scrape article headlines from the 3,815 provided URLs.
- Handle failed requests (non-200 status codes, changed page structures, paywalls) by logging and skipping.
- Supplement with additional self-scraped URLs from both outlets' recent article indexes to expand dataset size and recency diversity.
- Extract the `<h1>` headline tag (class `"headline speakable"` for Fox News; identify equivalent for NBC News).
- Store results in a CSV with columns: `headline`, `source`, `url`.

### 2.2 Data Cleaning & Preprocessing

| Step | Description |
|------|-------------|
| Deduplication | Remove exact-duplicate headlines |
| Missing data | Drop rows where headline is null or empty |
| HTML artifacts | Strip residual HTML tags, special characters, non-ASCII |
| Normalization | Lowercase; strip leading/trailing whitespace |
| Stopword removal | Optionally remove stopwords for classical models; skip for transformers |
| Stemming/Lemmatization | Apply for TF-IDF-based models; skip for BERT-based models |
| Train/Val/Test split | 70% train / 15% val / 15% internal test (stratified by source) |
| Class balance check | Verify label distribution; apply oversampling/class weights if skewed |

### 2.3 Model Design

We plan an incremental model progression, using each as a baseline for the next:

#### Stage 1 — Reproduce Baseline
- TF-IDF (max 100 features) + Logistic Regression  
- Target: ~66.49% accuracy (as given)

#### Stage 2 — Classical ML Improvements
- TF-IDF with expanded vocabulary (max 5,000–10,000 features), bigrams
- Models: Logistic Regression (tuned C), SVM (linear kernel), Gradient Boosting (XGBoost/LightGBM)
- Feature engineering: headline length, capitalization ratio, named entity density (via spaCy)

#### Stage 3 — Neural Text Models
- Fine-tune a pre-trained transformer: **DistilBERT** or **RoBERTa** (via HuggingFace `transformers`)
- Add a classification head; fine-tune with AdamW, learning rate ~2e-5, 3–5 epochs
- Evaluate on internal validation set; use early stopping

#### Stage 4 (Exploratory) — Analysis & Ablations
- Error analysis: qualitative review of misclassified headlines
- Ablation study: impact of dataset size, feature types, and text normalization choices
- Investigate model confidence calibration and headline-length effects

### 2.4 Evaluation Protocol

- **Primary metric:** Classification accuracy on the course leaderboard (held-out post-deadline scrape)
- **Reported metrics:** Accuracy, Precision, Recall, F1-score (macro and per-class)
- **Internal evaluation:** Stratified held-out test set (15%); cross-validation on training set for hyperparameter tuning
- **Leaderboard:** Submit at least one entry; iterate to improve ranking

---

## 3. Exploratory Component

We will pursue an **Analysis** exploratory component:

> **Research Question:** What linguistic features most distinguish Fox News from NBC News headlines — and does this signal shift over time or across topic categories?

- Extract top TF-IDF tokens per class; visualize as word clouds and bar charts
- Use SHAP values on the fine-tuned transformer to identify which tokens drive predictions
- Segment headlines by topic (politics, sports, entertainment) and measure per-topic accuracy
- Examine whether the model's confidence correlates with article recency

This analysis will be included in the final report with visualizations and summary statistics.

---

## 4. Team Division of Labor

| Role | Member | GitHub | Responsibilities |
|------|--------|--------|-----------------|
| **Data Engineer** | Steven Gavin Sears | [@Gavin-Sears](https://github.com/Gavin-Sears) | Write and run the web scraper for all 3,815+ URLs; handle failures and retries; implement cleaning pipeline; produce final `merged_news_data.csv`; document data statistics (size, class balance, headline length distribution) |
| **Model Trainer** | Dawei Liu | [@davidliuk](https://github.com/davidliuk) | Reproduce baseline; implement Stage 2 classical models; fine-tune DistilBERT/RoBERTa; manage hyperparameter search; submit to leaderboard; maintain experiment logs (accuracy, F1 per run) |
| **Analysis & Report** | Johnny Ding | [@JohnnySIST](https://github.com/JohnnySIST) | Conduct error analysis and ablation studies; generate figures (line charts of model metrics, word clouds, SHAP plots); write all sections of the 5-page project report; coordinate final submission packaging |

> **Note:** All members are expected to participate in design discussions and code review. Gavin hands off clean data to Dawei; Dawei provides model outputs and logs to Johnny.

---

## 5. Milestones

| Date | Milestone | Owner |
|------|-----------|-------|
| **Apr 13** | Gradescope team formation submitted | All |
| **Apr 16** | Scraper complete; raw headlines CSV collected | A |
| **Apr 19** | Cleaning pipeline done; final train/val/test split ready | A |
| **Apr 21** | Baseline reproduced (66.49%); Stage 2 models running | B |
| **Apr 25** | First leaderboard submission | B |
| **Apr 27** | DistilBERT fine-tuning complete; best model selected | B |
| **Apr 28** | Error analysis and SHAP analysis complete; figures drafted | C |
| **May 1** | Report first draft complete; internal review | All |
| **May 4** | Report finalized; all submission artifacts packaged | All |
| **May 6** | Final submission on Gradescope (dataset + model + report) | All |

---

## 6. Deliverables (per course requirements)

- [ ] **Dataset:** `merged_news_data.csv` with scraped and cleaned headlines + source labels
- [ ] **Best model:** Fine-tuned transformer checkpoint (or sklearn pipeline pickle for classical model)
- [ ] **Project report:** 5-page report covering data collection, model design, evaluation, and exploratory analysis
- [ ] **Leaderboard entry:** At least one submission before deadline

---

## 7. Tools & Infrastructure

| Component | Tool |
|-----------|------|
| Web scraping | `requests`, `BeautifulSoup4` |
| Data processing | `pandas`, `numpy`, `scikit-learn` |
| Classical ML | `scikit-learn`, `xgboost` |
| Transformer fine-tuning | `transformers` (HuggingFace), `torch` |
| NLP utilities | `spaCy`, `nltk` |
| Experiment tracking | Google Colab + manual logs (CSV/spreadsheet) |
| Version control | Git (shared repository) |
| Interpretability | `shap`, `matplotlib`, `seaborn` |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Scraping failures (paywalls, 404s) | Log all failures; supplement with self-collected URLs |
| Class imbalance (2010 Fox vs 1805 NBC) | Use stratified splits; report per-class F1 |
| GPU quota limits on Colab | Use DistilBERT (smaller than BERT-base); cache tokenized inputs |
| Overfitting to training URLs | Evaluate on post-deadline scrape — keep val/test split clean |
| Deadline risk | All code paths independently testable per member's responsibility |
