# CTG Benchmarking: Random Forest vs 1D-CNN for Fetal Heart Rate Pattern Detection

This repository contains the implementation and outputs for an FYP benchmarking study on machine-learning classifiers for cardiotocography (CTG) fetal heart rate pattern detection.

The study is framed as a controlled, reproducible benchmark (not a production real-time system), comparing:

- Random Forest (feature-based baseline)
- 1D-CNN (raw-signal representation model)

under shared preprocessing, leakage-aware grouped splitting, and common evaluation criteria.

## Overview

Cardiotocography (CTG) is widely used for intrapartum fetal monitoring, but interpretation can be subjective and variable.
This project benchmarks model behavior under controlled conditions to support transparent decision-support development.

The benchmarking motivation was shaped during Residency 4 context at the INFANT Research Centre.

## Scope and Claim Boundary

- Dataset: CTU-UHB Intrapartum Cardiotocography Database (PhysioNet)
- Current benchmark scope: within-cohort evaluation
- Label regime: weakly supervised windows derived from record-level outcomes
- Evaluation framing: fixed-split comparative evidence
- Not claimed in this dissertation run: repeated grouped resampling inferential statistics

## Main Result (Current Benchmark Cycle)

On held-out windows:

- Random Forest: 95.55% accuracy, macro-F1 0.945
- 1D-CNN: 65.31% accuracy, macro-F1 0.584

At record level (CNN), performance drops to:

- 48.19% accuracy, macro-F1 0.383

Under this data regime, Random Forest is the stronger current transparent baseline.

## Repository Structure

```text
CTG/
├── data/
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   ├── cnn.py
│   └── utils.py
├── notebooks/
│   ├── 01-BlockA-DataDiscovery.ipynb
│   ├── 02-BlockB-Preprocessing.ipynb
│   ├── 03-BlockC-Segmentation.ipynb
│   ├── 04-BlockD-FeatureEngineering.ipynb
│   ├── 05-BlockE-RandomForest.ipynb
│   ├── 06-BlockF-PostProcessing.ipynb
│   └── 07-BlockG-CNN.ipynb
├── outputs/
│   └── models/
├── doc/
│   ├── thesis.typ
│   └── thesis.pdf
├── pyproject.toml
└── README.md
```

## Environment

Primary runtime used in dissertation runs (see thesis appendix for full table):

- Python 3.10.19
- Linux
- CPU execution
- Core libs: numpy, pandas, scipy, scikit-learn, wfdb, matplotlib, seaborn, joblib
- CNN path: torch

## Installation

```bash
git clone https://github.com/Naem-Haq/CTG.git
cd CTG
pip install -e .
```

Optional CNN dependency:

```bash
pip install torch
```

## Reproducible Run Sequence

Core pipeline modules:

```bash
python -m src.preprocessing
python -m src.segmentation
python -m src.feature_extraction
```

Notebook execution for benchmark artifacts:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/05-BlockE-RandomForest.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute notebooks/06-BlockF-PostProcessing.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute notebooks/07-BlockG-CNN.ipynb --inplace
```

Compile dissertation PDF:

```bash
typst compile --root /absolute/path/to/CTG doc/thesis.typ doc/thesis.pdf
```

## Key Artifacts

Generated under `outputs/models/` (examples):

- `rf_3class_report.json`
- `cnn_3class_report.json`
- `cnn_vs_rf_comparison.json`
- `rf_3class_confusion_matrix.png`
- `cnn_3class_confusion_matrix.png`
- `rf_3class_feature_importance.png`
- `blockF_alert_timeline.png`

## Limitations and Future Work

Current findings are bounded to CTU-UHB and this weak-label setting.
Priority extensions:

- repeated grouped resampling and stronger uncertainty analysis
- external validation across broader datasets
- prospective clinician-facing translation workflows

## References

- CTU-UHB Database (PhysioNet): https://physionet.org/content/ctu-uhb-ctgdb/
- Thesis source: `doc/thesis.typ`
- Thesis PDF: `doc/thesis.pdf`
