# Communication Churn Prediction in Multichannel CRM Platforms

Machine Learning pipeline to predict which users will stop interacting with multichannel messaging campaigns (email and mobile push) at a mid-sized e-commerce retailer.

Paper: [_"Applying Machine Learning to Communication Churn Prediction in Multichannel CRM Platforms: Email, Push, and In-App"_](Communication-Churn-Prediction-in-Multichannel-CRM-Platforms.pdf) — Trenyan, A. (2026).

---

## Results

Evaluation over **~1.98M users** (window 2022-10-23 → 2023-04-23), with 95% bootstrap confidence intervals (200 iterations):

| Model | AUC-ROC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.771 | 0.773 | 0.791 | 0.675 | 0.954 |
| **Random Forest** | **0.807** | **0.821** | **0.804** | **0.701** | **0.943** |
| XGBoost | 0.806 | 0.820 | 0.803 | 0.698 | 0.946 |

**Winning model: Random Forest** (best AUC-ROC and PR-AUC).

Optimal threshold selected by maximum F1 on the test set. See the full report in [`outputs/report.md`](outputs/report.md).

### ROC and Precision-Recall curves

![ROC curves](outputs/figures/roc_curves_20260519_152108.png)

![PR curves](outputs/figures/pr_curves_20260519_152108.png)

### Feature importance — Random Forest

![Feature importance RF](outputs/figures/feature_importance_rf_20260519_152108.png)

### Confusion matrices

| Logistic Regression | Random Forest | XGBoost |
|---|---|---|
| ![LR](outputs/figures/confusion_logreg_20260519_152108.png) | ![RF](outputs/figures/confusion_rf_20260519_152108.png) | ![XGB](outputs/figures/confusion_xgb_20260519_152108.png) |

---

## Repository structure

```
ml-crm-churn/
├── run.py                  # Pipeline entrypoint
├── config.json             # Hyperparameters and global configuration
├── requirements.txt        # Python dependencies
├── referencias.bib         # BibTeX bibliography of the paper
├── Communication-Churn-Prediction-in-Multichannel-CRM-Platforms.pdf
├── src/
│   ├── data.py             # Phase 1: loading and validation
│   ├── features.py         # Phase 2: feature engineering
│   ├── train.py            # Phase 3: training (LR, RF, XGB)
│   ├── evaluate.py         # Phase 4: evaluation and figures
│   └── logging_utils.py    # Shared logger and helpers
├── data/                   # ← empty in the repo; populate with the dataset
└── outputs/
    ├── figures/            # Generated figures (versioned)
    ├── report.md           # Results report (versioned)
    ├── models/             # .pkl models (not versioned, regenerated)
    ├── features/           # .parquet features (not versioned)
    └── logs/               # Run logs (not versioned)
```

---

## Environment setup

**Requirements**: Python 3.11.x

```bash
# Create virtual environment and install dependencies
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset download

The pipeline requires 4 files in the `data/` folder:

### Auxiliary files (Kaggle)

Download the [**Direct Messaging**](https://www.kaggle.com/datasets/mkechinov/direct-messaging) dataset from Kaggle and copy the following files to `data/`:

- `campaigns.csv`
- `holidays.csv`
- `client_first_purchase_date.csv`

```bash
# With the Kaggle CLI (pip install kaggle)
kaggle datasets download mkechinov/direct-messaging --unzip -p data/
```

### Full messages dataset

`messages.csv.gz` (21.5 GB compressed, ~721M rows) is downloaded directly from REES46:

```bash
wget -P data/ https://data.rees46.com/datasets/direct-messaging/messages.csv.gz
```

> **Note**: The download is ~21.5 GB. The pipeline processes the compressed file via streaming (without decompressing to disk), so ~30 GB of free space is required in total.

**Expected MD5** of the full file: `95fa332ef970a50c6c18a916b79f99af`

```bash
# Verify integrity
md5sum data/messages.csv.gz   # Linux
md5 data/messages.csv.gz      # macOS
```

---

## Running the pipeline

The pipeline has 4 phases that can be run together or individually:

```bash
# Run all 4 phases in order (recommended the first time)
python run.py

# Run a single phase
python run.py --phase data        # Phase 1: dataset validation
python run.py --phase features    # Phase 2: feature engineering (~9h on CPU)
python run.py --phase train       # Phase 3: model training
python run.py --phase evaluate    # Phase 4: evaluation and figure generation

# Force feature recomputation (ignores cache)
python run.py --phase features --force
```

### Estimated runtimes (CPU, no GPU)

| Phase | Approximate time |
|---|---|
| Phase 1 — validation | ~30–60 min (MD5 read + streaming) |
| Phase 2 — features | ~9–12 h (processing 721M rows) |
| Phase 3 — training | ~2–4 h (hyperparameter search + refit) |
| Phase 4 — evaluation | ~15–30 min (bootstrap, 200 iters) |

> Phases 3 and 4 generate timestamped artifacts in `outputs/models/` and `outputs/figures/`.

### Configuration

Pipeline parameters are controlled from `config.json`:

| Parameter | Default value | Description |
|---|---|---|
| `seed` | 42 | Global random seed |
| `ventana_observacion_meses` | 18 | Months of history for the features |
| `ventana_evaluacion_meses` | 6 | Prediction horizon |
| `smote_threshold` | 0.2 | Imbalance threshold to apply SMOTE |
| `random_search_n_iter` | 20 | RandomizedSearchCV iterations |
| `cv_folds` | 5 | Cross-validation folds |
| `bootstrap_iters` | 200 | Iterations for 95% CI |
| `chunksize_messages` | 500 000 | Rows per chunk when reading `messages.csv.gz` |
| `subsample_busqueda` | 500 000 | Subsample for hyperparameter search |

---

## Methodology (summary)

- **Target**: communication churn = a user who neither opens nor clicks any email or mobile push message during the evaluation window (6 months).
- **Features**: 40+ variables per user (open, click, bounce and unsubscribe rates, sending frequency, recency, cross-channel activity ratio, tenure, etc.).
- **Split**: strict temporal split (features from the observation window, target from the evaluation window).
- **Models**: Logistic Regression (baseline), Random Forest, XGBoost.
- **Evaluation**: AUC-ROC, PR-AUC, F1, Precision, Recall with optimal threshold and bootstrapping.

---

## Tech stack

| Library | Version |
|---|---|
| Python | 3.11.4 |
| pandas | 3.0.3 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| imbalanced-learn | 0.14.1 |
| matplotlib | 3.10.9 |
| pyarrow | 24.0.0 |
