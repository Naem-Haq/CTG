# AI4Life: Cardiotocography Decision Support System

A biomedical signal processing and machine learning project to develop an **AI-based decision support system** for monitoring fetal wellbeing during labour using Cardiotocography (CTG) data.

## 📋 Overview

**Cardiotocography (CTG)** measures fetal and maternal heart rate and uterine contractions during labour. CTG interpretation is subjective and controversial, with significant inter-observer variability. This project applies machine learning to create a reliable, interpretable decision support tool to assist clinicians in identifying fetal distress and optimizing clinical interventions.

### Key Objective
Develop a **real-time decision support system** that:
- Analyzes fetal heart rate (FHR), maternal heart rate (MHR), and uterine contraction (UC) signals
- Automatically identifies clinically significant changes indicating potential fetal distress
- Provides interpretable, temporal alerts to support clinical decision-making
- Reduces unnecessary interventions while improving maternal and neonatal outcomes

---

## 🎯 Project Goals

✅ Extract meaningful physiological features from raw CTG signals  
✅ Build interpretable machine learning models (Random Forest baseline, CNN advanced)  
✅ Validate model performance on CTU-CHB database (552 recordings)  
✅ Implement post-processing for clinical alert generation  
✅ Document limitations and clinical implementation requirements  

---

## 📊 Dataset

**CTU-CHB Intrapartum Cardiotocography Database** (PhysioNet)
- **552 recordings** from Czech University Hospital Brno
- **~6500 hours** of continuous monitoring
- **Signals:** Fetal Heart Rate (FHR), Uterine Contractions (UC), Maternal Heart Rate (MHR)
- **Sampling rate:** 4 Hz
- **Duration:** ~90 minutes per recording
- **Labels:** Normal vs. Pathological classification + clinical outcomes (pH, Apgar scores)

[Download database](https://physionet.org/content/ctu-chb-intrapartum-cardiotocography-database-1.0.0/)

---

## 🏗️ Project Structure

```
CTG/
├── notebooks/
│   ├── 01-BlockA-DataDiscovery.ipynb         # Data validation & summary
│   ├── 02-BlockB-Preprocessing.ipynb         # Signal cleaning pipeline
│   ├── 03-BlockC-Segmentation.ipynb          # Window-based segmentation
│   ├── 04-BlockD-FeatureEngineering.ipynb    # Clinical feature extraction
│   ├── 05-BlockE-RandomForest.ipynb          # Baseline RF classifier
│   ├── 06-BlockF-PostProcessing.ipynb        # Alert generation
│   └── 07-BlockG-CNN.ipynb                   # Advanced CNN model (optional)
├── src/
│   ├── preprocessing.py                       # Reusable preprocessing functions
│   ├── segmentation.py                        # Segmentation utilities
│   ├── feature_extraction.py                  # Feature computation
│   └── utils.py                               # Common utilities
├── outputs/
│   ├── dataset_summary.csv                    # Block A deliverable
│   ├── dataset_statistics.json                # Block A statistics
│   └── models/                                # Trained models (RF, CNN)
├── data/
│   └── ctu-chb-intrapartum-cardiotocography-database-1.0.0/
│       └── (552 .hea/.dat files)
├── plan.md                                    # Detailed project plan
└── README.md                                  # This file
```

---

## 🚀 Implementation Blocks

### **Block A: Data Discovery & Validation** ✅ COMPLETE
Scan all 552 recordings to understand:
- Signal availability (FHR, UC, MHR)
- Recording durations and sampling rates
- Outcome label distribution (Normal vs. Pathological)
- Data quality metrics (missing data, outliers)

**Output:** `dataset_summary.csv`, `dataset_statistics.json`

### **Block B: Preprocessing Pipeline**
Clean signals by:
- Detecting and interpolating signal dropouts
- Removing physiological outliers (FHR outside [80, 240] bpm)
- Applying Gaussian smoothing
- Per-recording z-score normalization

**Output:** Cleaned FHR and UC signals

### **Block C: Segmentation Strategy**
Convert 90-minute recordings into fixed-length windows:
- Window size: 5 minutes (configurable)
- Non-overlapping segments
- Label assignment: All segments inherit global recording label

**Output:** Segmentation manifest CSV

### **Block D: Feature Engineering**
Extract 50+ clinical features:
- **Baseline:** Mean FHR, Median FHR
- **Variability:** Short-term/long-term variability, std dev
- **Decelerations:** Count, depth, duration, recovery slope
- **Accelerations:** Count, amplitude, rise time
- **Frequency-domain:** Peak frequencies, spectral power
- **Entropy:** Sample entropy, complexity measures
- **UC Features:** Amplitude, frequency, baseline tonus

**Output:** Feature matrix (N_segments × N_features)

### **Block E: Baseline Model – Random Forest**
Train explainable classifier:
- 100–200 trees, max_depth=10, balanced class weights
- 70/30 train/test split (stratified)
- Metrics: ROC-AUC, Sensitivity, Specificity, Confusion Matrix
- Feature importance ranking

**Output:** Trained RF model, performance report, ROC curve

### **Block F: Post-Processing & Temporal Smoothing**
Generate clinician-friendly alerts:
- Recreate 3-class NICE pseudo-ground-truth labels from feature matrix
- Evaluate alerts on grouped holdout records (same no-leakage split policy as Block E)
- Apply causal temporal smoothing (trailing majority vote per record)
- Report clinical alert metrics (false alerts/hour, transition rate, escalation rate)

**Output:** Real-time alert timeline, visualization

### **Block G: Advanced Model – CNN** (Optional)
Learn features directly from raw signals:
- 1D CNN on segmented FHR + UC
- Compare CNN vs. RF (accuracy, interpretability, speed)
- Visualize learned patterns

**Output:** CNN model, side-by-side comparison

---

## 📦 Installation & Setup

### Requirements
- Python 3.8+
- NumPy, Pandas, Scikit-learn, SciPy, Matplotlib
- PhysioNet WFDB library
- TensorFlow/Keras (optional, for CNN)

### Quick Start
```bash
# Clone repository
git clone https://github.com/Naem-Haq/CTG.git
cd CTG

# Install dependencies
pip install -e .

# Download dataset
# Visit: https://physionet.org/content/ctu-chb-intrapartum-cardiotocography-database-1.0.0/
# Extract to: data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/

# Run Block A (data discovery)
jupyter notebook notebooks/01-BlockA-DataDiscovery.ipynb
```

---

## 📈 Usage

1. **Data Validation:** Run Block A to generate dataset summary
2. **Preprocessing:** Run Block B to clean signals
3. **Segmentation:** Run Block C to create training windows
4. **Features:** Run Block D to compute feature matrix
5. **Training:** Run Block E to train Random Forest
6. **Deployment:** Run Block F to generate real-time alerts

---

## ⚠️ Important Assumptions

- **Weak Labels:** Outcome labels are recording-level. Some "normal" segments may contain transient stress.
- **Dataset Bias:** Czech cohort only; may not generalize globally.
- **Class Imbalance:** Majority of cases are normal; pathological cases are minority.
- **No Time-Localization:** Adverse events are not time-stamped within recordings.

**See `plan.md` for full methodology details.**

---

## 👤 Contact

**Supervisor:** Professor Liam Marnane  
**Email:** l.marnane@ucc.ie  
**Institution:** University College Cork, School of Engineering

---

## 📚 References

- PhysioNet CTU-CHB Database: https://physionet.org/content/ctu-chb-intrapartum-cardiotocography-database-1.0.0/
- FIGO CTG Guidelines: International Federation of Gynecology and Obstetrics
- Signal Processing in Biomedicine: Classic textbooks on biomedical signal processing

---

## 📝 License

This project is developed for academic research purposes.

---

**Last Updated:** January 27, 2026
