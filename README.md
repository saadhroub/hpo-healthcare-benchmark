# Hyperparameter Optimization for Healthcare Machine Learning

Reproducibility artefacts for the manuscript

> **An Experimental Comparison of Hyperparameter Optimization Methods for Healthcare Machine Learning: From Grid Search to Swarm Intelligence**
> Saadeddin Hroub, Masri Ayob, Mohd Nor Akmal Khalid
> Data Mining and Optimization Lab, Universiti Kebangsaan Malaysia
> Submitted to *[target journal]*, 2026.

This repository contains the code, cleaned datasets, raw per-seed results, summary tables, publication figures, and LaTeX source needed to reproduce every number, table, and figure in the paper.

---

## What's in the paper

A systematic head-to-head evaluation of six hyperparameter optimization (HPO) strategies — Manual tuning, Grid search, Random search, Bayesian optimization (Optuna/TPE), Grey Wolf Optimizer (GWO), and Particle Swarm Optimization (PSO) — across three classifiers (Random Forest, SVM with RBF kernel, Logistic Regression) on four UCI healthcare benchmarks:

| Dataset              | n   | Features | Positive class      |
|----------------------|-----|----------|---------------------|
| Breast Cancer Wisconsin | 569 | 30       | Malignant           |
| Heart Disease (Statlog) | 270 | 13       | Presence of disease |
| Pima Indians Diabetes   | 768 | 8        | Diabetic            |
| Indian Liver Patient    | 583 | 10       | Liver patient       |

Each (dataset, classifier, method) triple is run with ten independent seeds (42–51), giving 720 optimization runs. Pairwise significance against the manual baseline is tested with the Wilcoxon signed-rank test at α = 0.05, with Holm–Bonferroni correction over the 60 non-manual comparisons, and Cliff's δ as an effect-size measure.

### Headline finding

Across the 60 non-manual comparisons, GWO/PSO attain the numerically best accuracy in only one cell and produce **zero** Holm-corrected significant accuracy improvements over manual tuning. The only Holm-corrected significant departures are two accuracy *decreases* on ILPD/SVM — which coincide with a 15-percentage-point macro-F₁ recovery from a degenerate majority-class baseline and are clinically favourable. Random search is Pareto-dominant among optimizers on every dataset.

---

## Repository layout

```
.
├── README.md                         # this file
├── LICENSE                           # MIT
├── CITATION.cff                      # machine-readable citation metadata
├── requirements.txt                  # Python deps (pinned)
├── .gitignore
├── datasets/                         # cleaned CSVs + provenance
│   ├── README.md
│   ├── dataset_card.md               # dataset-by-dataset provenance
│   ├── dataset_manifest.json         # SHA256, row/col counts, class balance
│   ├── download_datasets.py          # re-fetches raw Pima + ILPD from UCI
│   ├── pima_diabetes_clean.csv
│   ├── ilpd_clean.csv
│   └── (BC, HD are loaded from sklearn.datasets / ucimlrepo at runtime)
├── notebooks/
│   ├── hpo_experiments.ipynb         # main benchmark (runs the 720 seeds)
│   └── figures_supplementary.ipynb   # instrumented GWO/PSO for convergence/diversity panels
├── src/
│   └── make_figures.py               # regenerates all paper figures from results CSVs
├── results/                          # committed outputs for verification
│   ├── results_raw.csv               # 720 rows, one per (seed, method, classifier, dataset)
│   ├── results_summary.csv           # 72 (dataset, classifier, method) cells with p, Holm p, Cliff's δ
│   └── latex_tables.tex              # pre-rendered LaTeX rows (used by the paper)
├── figures/                          # committed PDF + PNG
│   ├── time_accuracy_all.pdf/.png
│   ├── box_plots_all.pdf/.png
│   ├── ilpd_svm_story.pdf/.png
│   ├── summary_scatter.pdf/.png
│   └── meta_heuristic_pd_rf.pdf/.png
└── paper/
    └── root_journal.tex              # LaTeX source
    └── root_journal.pdf              # compiled manuscript
```

---

## Reproducing the results

### 1. Environment

```bash
git clone https://github.com/saadhroub/hpo-healthcare-benchmark.git
cd hpo-healthcare-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10 and the package versions pinned in `requirements.txt`:

- `scikit-learn==1.3.2`
- `optuna==3.4.0`
- `numpy>=1.24,<2.0` (NumPy 2.x breaks the Colab dependency chain — see CHANGELOG)
- `pandas>=2.0,<3.0`
- `scipy>=1.11,<1.14`
- `matplotlib>=3.7`
- `pdf2image`, `pyarrow` (optional, for figure re-rendering)

### 2. Datasets

- **Breast Cancer Wisconsin** is loaded at runtime from `sklearn.datasets.load_breast_cancer()`.
- **Heart Disease (Statlog)** is loaded at runtime via `ucimlrepo` (fetched from UCI ML Repository, id=145).
- **Pima Indians Diabetes** and **Indian Liver Patient Dataset** are included as cleaned CSVs in `datasets/` along with a `download_datasets.py` script that reproduces the cleaning pipeline from the original UCI/Kaggle raw files. SHA256 checksums are recorded in `dataset_manifest.json`.

### 3. Running the benchmark

Open `notebooks/hpo_experiments.ipynb` in Jupyter or Google Colab and run all cells. The notebook:

1. Downloads/loads all four datasets.
2. Runs the 4×3×6×10 = 720 HPO trials with fixed seeds (42–51) and per-block checkpointing so a disconnect only loses the current block.
3. Writes `results/results_raw.csv` (720 rows) and `results/results_summary.csv` (72 cells).
4. Emits `latex_tables.tex` for the paper.

Typical runtime on a free Colab instance: ~3 hours total (Grid search on Random Forest dominates). Per-dataset checkpointing means you can resume from whichever `*_checkpoint.csv` was last written.

### 4. Regenerating figures

```bash
python src/make_figures.py
```

This reads the CSVs in `results/` and writes all five publication PDFs into `figures/`. PNGs can be re-rendered with:

```bash
python -c "
from pdf2image import convert_from_path
import os
for fn in os.listdir('figures'):
    if fn.endswith('.pdf'):
        pages = convert_from_path(f'figures/{fn}', dpi=140)
        pages[0].save(f'figures/{fn[:-4]}.png', 'PNG')
"
```

### 5. Supplementary convergence/diversity panels

`notebooks/figures_supplementary.ipynb` contains instrumented GWO/PSO that log per-iteration best-so-far fitness and mean pairwise population distance. Running this notebook regenerates panels (a) and (b) of `meta_heuristic_analysis.pdf`. These panels are separate from the main benchmark and do not affect the tables.

### 6. Compiling the paper

```bash
cd paper
pdflatex root_journal.tex
pdflatex root_journal.tex   # second pass for cross-references
```

The default template is `ieeeconf`; swap `\documentclass` to `IEEEtran` (`journal` option) for IEEE Access, or to `elsarticle`/`svjour3` for the respective publishers.

---

## Data and code availability statement

All code, cleaned data, raw per-seed outputs, summary tables, and publication figures are released under the MIT license at this repository. The original raw datasets remain the property of their respective sources (UCI Machine Learning Repository for HD, Pima, ILPD; scikit-learn / University of Wisconsin for BC) and are redistributed here only in the cleaned, analysis-ready form necessary for reproducibility.

---

## Citation

If you use this code or results, please cite the paper:

```bibtex
@article{hroub2026hpo,
  author  = {Hroub, Saadeddin and Ayob, Masri and Khalid, Mohd Nor Akmal},
  title   = {An Experimental Comparison of Hyperparameter Optimization Methods
             for Healthcare Machine Learning: From Grid Search to Swarm Intelligence},
  journal = {[target journal]},
  year    = {2026},
  note    = {Code: https://github.com/saadhroub/hpo-healthcare-benchmark}
}
```

A `CITATION.cff` file is included for GitHub's "Cite this repository" feature.

---

## Acknowledgement

This work was supported by the Ministry of Higher Education Malaysia through the Transdisciplinary Research Grant Scheme (FRGS/1/2024/ICT02/UKM/01/) and carried out at the Data Mining and Optimization Lab, Faculty of Information Science and Technology, Universiti Kebangsaan Malaysia.

## Contact

Saadeddin Hroub — saad.hroub@gmail.com
Masri Ayob (corresponding author) — masri@ukm.edu.my
