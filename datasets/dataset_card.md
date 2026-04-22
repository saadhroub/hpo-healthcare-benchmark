# Dataset Card — Two Additional Healthcare Benchmarks

This card documents the two datasets added to the HPO study to move from a two-dataset to a four-dataset benchmark. It is written in the style expected by methods sections of IEEE Access / PeerJ CS / BMC MIDM, so text can be pasted into the manuscript with minimal editing.

---

## 1. Pima Indians Diabetes Database

### Provenance

**Origin.** The data were collected by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and were curated for ML use by Smith et al. (1988) [1]. All participants are females of Pima Indian heritage aged at least 21 years.

**Canonical sources (any of these work; the downloader tries them in order):**

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/34/diabetes
- Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- OpenML: https://www.openml.org/search?type=data&id=43483
- Direct CSV mirror (headerless): https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
- Direct CSV mirror (with header): https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv

**License.** CC BY 4.0 (UCI).

### Schema

| # | Column | Type | Description |
|---|---|---|---|
| 1 | Pregnancies | int | Number of times pregnant |
| 2 | Glucose | int | Plasma glucose concentration (2-hour OGTT, mg/dL) |
| 3 | BloodPressure | int | Diastolic blood pressure (mm Hg) |
| 4 | SkinThickness | int | Triceps skinfold thickness (mm) |
| 5 | Insulin | int | 2-hour serum insulin (μU/mL) |
| 6 | BMI | float | Body mass index (kg/m²) |
| 7 | DiabetesPedigreeFunction | float | Pedigree-based genetic risk score |
| 8 | Age | int | Age in years |
| 9 | **Outcome** | int (0/1) | 1 = diabetes onset within 5 years; 0 = no onset |

- **Rows:** 768
- **Class balance:** 500 negative / 268 positive (≈ 65 / 35)

### Known data-quality issues (important for the manuscript)

Biological zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are used by the original curators as the missing-value indicator. Any reviewer will ask how they are handled. The downloader script performs **median imputation** after replacing zeros with NaN. For a reviewer-defensible protocol, imputation medians should be computed **within each cross-validation training fold** rather than on the full dataset, and the manuscript should state this explicitly. The `download_datasets.py` script imputes globally for convenience; replace with fold-level imputation in the main experiment driver.

### Methods-section text (copy into manuscript)

> The Pima Indians Diabetes dataset [1] contains 768 female participants (≥21 years, Pima heritage) with 8 clinical features (plasma glucose, diastolic blood pressure, triceps skinfold thickness, serum insulin, BMI, diabetes pedigree, age, number of pregnancies). The target is onset of diabetes within five years (268 positive, 500 negative). Following standard preprocessing [2], biologically-implausible zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI were replaced with missing indicators and imputed with the per-fold training-set median to prevent leakage.

### Citations to add to bibliography

- [1] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, R. S. Johannes, "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus," *Proc. Symp. Comp. Appl. Med. Care*, pp. 261–265, 1988.
- [2] D. Dua and C. Graff, *UCI Machine Learning Repository*, Univ. California, Irvine, 2019. [Online]. Available: https://archive.ics.uci.edu/ml

---

## 2. Indian Liver Patient Dataset (ILPD)

### Provenance

**Origin.** Collected from North East Andhra Pradesh, India, and donated to UCI by Ramana et al. (2012) [3]. The dataset was built to study liver-disease risk stratification from routine blood-panel lab results.

**Canonical sources:**

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
- Mirror with header: https://raw.githubusercontent.com/aniruddhachoudhury/Indian-Liver-Patient-Dataset/master/indian_liver_patient.csv

**License.** CC BY 4.0 (UCI).

### Schema (after our preprocessing; raw file is headerless)

| # | Column | Type | Description |
|---|---|---|---|
| 1 | Age | int | Age in years |
| 2 | Gender | int (0/1) | 1 = Male, 0 = Female |
| 3 | TotalBilirubin | float | Total bilirubin (mg/dL) |
| 4 | DirectBilirubin | float | Direct bilirubin (mg/dL) |
| 5 | AlkalinePhosphatase | int | ALP (IU/L) |
| 6 | Alamine_Aminotransferase | int | ALT (IU/L) |
| 7 | Aspartate_Aminotransferase | int | AST (IU/L) |
| 8 | TotalProtein | float | Total protein (g/dL) |
| 9 | Albumin | float | Albumin (g/dL) |
| 10 | AG_Ratio | float | Albumin/Globulin ratio |
| 11 | **Outcome** | int (0/1) | 1 = liver patient, 0 = non-patient |

- **Rows:** 583
- **Class balance:** 167 negative / 416 positive (≈ 29 / 71) — **imbalanced**

### Known data-quality issues

- **Target recoding.** The UCI raw file codes 1 = liver patient, 2 = non-patient. We recode to the conventional {1 = disease present, 0 = disease absent}.
- **Gender encoding.** Raw is the string "Male" / "Female"; we encode as {Male = 1, Female = 0}.
- **Missingness.** Four rows have missing AG_Ratio. The downloader imputes with the dataset median; for formal evaluation, use per-fold median within training folds.
- **Class imbalance.** Positive class is majority. Report macro-F1 and per-class F1 separately; single-threshold F1 on the positive class alone can be misleading. The paper's existing FNR column handles this naturally.

### Methods-section text (copy into manuscript)

> The Indian Liver Patient Dataset (ILPD) [3] contains 583 patient records from North East Andhra Pradesh, India, with 10 features: age, gender, total and direct bilirubin, alkaline phosphatase, alanine and aspartate aminotransferases, total protein, albumin, and albumin/globulin ratio. The original two-class label (1 = liver patient, 2 = non-patient) was recoded to {1 = disease present, 0 = disease absent}; Gender was one-hot encoded as a single binary column. Four rows with missing albumin/globulin ratio were imputed with the per-fold training-set median. The class distribution is 416 positive / 167 negative, so macro-F1, per-class F1, and FNR are reported alongside accuracy.

### Citations to add to bibliography

- [3] B. V. Ramana, M. S. P. Babu, and N. B. Venkateswarlu, "A critical study of selected classification algorithms for liver disease diagnosis," *International Journal of Database Management Systems*, vol. 3, no. 2, pp. 101–114, 2011.

---

## Four-dataset summary (for manuscript Table 1)

| Dataset | Rows | Features | Class balance (pos/neg) | Clinical domain | Feature type |
|---|---|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | 569 | 30 | 212 / 357 | Oncology | FNA cytology |
| Heart Disease Statlog | 270 | 13 | 120 / 150 | Cardiology | Clinical history + vitals |
| **Pima Indians Diabetes** | **768** | **8** | **268 / 500** | **Endocrinology** | **Clinical measurements** |
| **ILPD (liver)** | **583** | **10** | **416 / 167** | **Hepatology** | **Lab panel** |

Four disease areas, four feature regimes, all binary-classification — exactly the diversity profile reviewers expect from an empirical HPO benchmark.

---

## How to use

1. Open a terminal in the `datasets/` folder.
2. Install dependencies: `pip install pandas numpy requests`
3. Run: `python download_datasets.py`

The script writes `pima_diabetes_clean.csv`, `ilpd_clean.csv`, their raw counterparts, and a `dataset_manifest.json` containing SHA-256 hashes for reproducibility. Point your existing HPO pipeline at the `_clean.csv` files.

If one of the mirror URLs fails (e.g., UCI maintenance window), the script tries the next source automatically. If all sources fail, download the CSV manually from any URL above and place it in the `datasets/` folder; the script's preprocessing functions (`preprocess_pima`, `preprocess_ilpd`) are importable and will accept the bytes directly.
