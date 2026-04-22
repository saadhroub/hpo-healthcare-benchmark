# datasets/ — how to get the two new benchmarks

## One-command setup

Open a terminal in **this** folder and run:

```bash
pip install pandas numpy requests
python download_datasets.py
```

This produces:

- `pima_diabetes_raw.csv` / `pima_diabetes_clean.csv`
- `ilpd_raw.csv` / `ilpd_clean.csv`
- `dataset_manifest.json` (provenance + SHA-256 hashes)

Feed the `_clean.csv` files into your existing HPO pipeline — same schema expectations as Wisconsin/Statlog (features in columns 1..n-1, binary target in last column named `Outcome`).

## If automated download fails

Some university networks block raw GitHub and UCI. In that case, manually download one CSV per dataset from any of these:

### Pima Indians Diabetes

- https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database (Kaggle login required)
- https://archive.ics.uci.edu/dataset/34/diabetes
- https://www.openml.org/search?type=data&id=43483

Place the file in this folder (any filename is fine), then edit the bottom of `download_datasets.py` to read your local file and pass its bytes to `preprocess_pima()`.

### Indian Liver Patient Dataset (ILPD)

- https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
- https://www.openml.org/search?type=data&id=1480

Same drop-in process.

## Why the preprocessing matters

Both datasets have well-known issues that reviewers will ask about:

- **Pima:** zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI are missing-value flags, not real measurements. The script replaces them with NaN and imputes with the median.
- **ILPD:** original target is `1 = liver patient, 2 = non-patient`. The script recodes to `1 = disease, 0 = no disease` so it matches the convention used for Wisconsin/Statlog. Gender is encoded as 0/1.

**Important reproducibility note.** The script performs *dataset-wide* median imputation, which is fine for a quick smoke test but causes slight leakage in K-fold CV. In your experiment driver, move the imputation *inside* each training fold. The manuscript should state this explicitly — reviewers will look for it.

## What these give the paper

You move from a 2-dataset benchmark to a 4-dataset benchmark covering four disease areas:

- Oncology (Wisconsin)
- Cardiology (Statlog)
- Endocrinology (Pima)
- Hepatology (ILPD)

That is the smallest scope that plausibly gets through IEEE Access or PeerJ CS review.

See `dataset_card.md` for full provenance, schema, licensing, and ready-to-paste methods-section text.
