# Welding Defect Detection — Model Training Pipeline

## Folder Structure

```
welding_defect_detection/
├── requirements.txt
├── 01_download_dataset.py   ← Step 1: download + organize dataset
├── 02_train_model.py        ← Step 2: extract features + train models
├── 03_predict.py            ← Step 3: test on a single image
├── feature_extractor.py     ← Core module (used by training + API)
├── data/
│   ├── raw/                 ← Downloaded Kaggle data
│   └── processed/           ← Organized images by class
├── models/                  ← Saved models (created after training)
│   ├── binary_classifier.joblib
│   ├── defect_classifier.joblib
│   ├── scaler.joblib
│   └── metadata.json
└── results/                 ← Reports, confusion matrices, plots
```

---

## Setup (do this once)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Step 1 — Download the Dataset

1. Go to https://www.kaggle.com and create an account (free)
2. Accept the dataset terms: https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083
3. Go to: https://www.kaggle.com/settings → API → **Create New Token**
4. This downloads `kaggle.json`. Place it here:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<YourName>\.kaggle\kaggle.json`
5. Run:
```bash
python 01_download_dataset.py
```

This will download and organize the dataset into `data/processed/`.

---

## Step 2 — Train the Models

```bash
python 02_train_model.py
```

This will:
- Extract features from all images (takes ~5–15 min depending on dataset size)
- Train SVM, Random Forest, and KNN for both tasks
- Print a comparison table and pick the best model
- Save models to `models/`
- Save confusion matrices and feature importance plots to `results/`

**First run caches features** to `results/features_cache.csv` — subsequent runs are fast.

---

## Step 3 — Test on a Single Image

```bash
# Basic prediction
python 03_predict.py --image path/to/weld.jpg

# With full pipeline visualization
python 03_predict.py --image path/to/weld.jpg --visualize
```

---

## What the models output

| Field | Description |
|---|---|
| `status` | `good` or `bad` |
| `defect_type` | `crack`, `porosity`, `undercut`, `lack_of_fusion`, etc. |
| `confidence` | Probability per class |
| `features` | All 18 extracted feature values |

---

## Features Extracted

| Group | Features |
|---|---|
| GLCM Texture | contrast, energy, homogeneity, correlation, dissimilarity, ASM |
| LBP | lbp_mean, lbp_std, lbp_energy |
| Shape | area, perimeter, aspect_ratio, circularity, edge_density, solidity |
| Intensity Stats | mean_intensity, std_intensity, skewness |

---

## Next Steps (Phase 2 — Backend API)

Once models are trained, we'll build:
- **FastAPI server** with a `/predict` endpoint
- **SQLite database** storing results (weld_images + extracted_features tables)
- **Analytics dashboard** showing defect statistics over time