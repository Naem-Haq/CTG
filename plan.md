# Biomedical Signal Processing and Machine Learning to Provide Decision Support to Clinicians in the Use of CTG

**Supervisor:** Professor Liam Marnane
**Email:** l.marnane@ucc.ie

## Research Question(s)

The INFANT research centre and the School of Engineering is developing **AI4Life**, a novel AI-based system to monitor the vital signs of mother and baby during labour to quickly identify any issues.

**AI4LIFE** is a decision support tool to optimise the use of Cardiotocography (CTG) for monitoring fetal wellbeing during labour and childbirth, allowing more informed decisions on interventions and reducing neonatal and maternal morbidity and mortality.

**Cardiotocography (CTG)** measures fetal and maternal heart rate and uterine contractions during labour. CTG interpretation is a controversial topic with contention mainly focused around the categorisation of heart rate decelerations.

## Proposed Methods

The approach adopted will involve:

- **Signal preprocessing** to identify and remove unwanted artefacts such as movement and maternal heart rate artefacts
- **Develop signal processing features** that identify when possible changes which occur in the fetal heart rate
- **Train a Machine Learning Classifier** to identify these changes
- **Develop post processing methods** that utilise the output of the classifier to alert clinical staff

## Special Requirements

Data from the [PhysioNet CTG database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) and Cork University Maternity Hospital will be used in this project.

## Expected Results

The algorithms for the analysis and signal processing will be developed in Python.

---

## Project Goals and Objectives

The goal of this project is to develop a **real-time decision support system for clinicians** by applying biomedical signal processing and machine learning techniques to Cardiotocography (CTG) data.

The system will analyse fetal heart rate, maternal heart rate, and uterine contraction signals to automatically identify clinically significant changes that may indicate fetal distress during labour.

Given the absence of clear, time-localised indicators of adverse events in CTG data, the project will focus on:
- Extracting meaningful physiological features
- Learning patterns associated with abnormal or deteriorating fetal conditions
- Presenting interpretable alerts to support timely and informed clinical decision-making

Ultimately, this work aims to improve the reliability and consistency of CTG interpretation, reduce unnecessary interventions, and contribute to improved maternal and neonatal outcomes.

---

## Step-by-Step Plan (Realistic & Defensible)

This plan is designed specifically for the "no obvious pointers" problem — which is the core difficulty of CTG analysis.

---

## Implementation Blocks: Concrete, Actionable Tasks

The following blocks provide a detailed, implementation-focused breakdown. Each block has specific inputs, processes, and deliverables.

### Block A: Data Discovery & Validation

**Objective:** Understand the exact structure, availability, and quality of the CTU-CHB dataset before any processing.

**What to Identify:**
- **Signal inventory:** Parse all 552 `.hea` files to confirm:
  - FHR (Fetal Heart Rate) availability and format
  - UC (Uterine Contraction) signal availability
  - Sampling rates (expect 4 Hz for both)
  - Recording durations
- **Outcome labels available:**
  - pH (umbilical artery blood sample)
  - Apgar score (1-min and 5-min)
  - Neonatal outcomes (seizures, NICU admission, HIE)
  - Pathological vs. normal classification
- **Data quality metrics:**
  - Missing signal segments (dropouts)
  - Signal quality (percentage of valid FHR data per 30-min window, as per inclusion criteria: >50%)
  - Sampling consistency
- **Class distribution:** Count normal vs. pathological cases to understand baseline imbalance

**Deliverable:**
A `dataset_summary.py` script that generates a summary report (CSV or JSON) containing:
- Total recordings analyzed
- Mean/median recording duration
- Percentage with pH available
- Number of normal cases
- Number of pathological cases
- Signal quality distribution
- Missing data statistics

---

### Block B: Preprocessing Pipeline

**Objective:** Create modular, reusable signal cleaning functions. Artefacts must be removed so the ML model learns physiological signals, not noise.

**What You Need:**
- **Raw data loader:**
  - Use PhysioNet WFDB library to read `.dat` files
  - Parse metadata from `.hea` files
  - Return time-series as NumPy arrays
- **Quality checks:**
  - Detect signal dropout periods
  - Flag segments with consistently poor signal quality
- **Artifact removal module:**
  - **Gaussian smoothing** (preserve FHR variability, avoid over-smoothing)
  - **Interpolation** for short gaps (<30 seconds)
  - **Outlier clipping:** Remove physiologically implausible values
    - FHR outside [80, 240] bpm
    - UC baseline drift correction
  - **Maternal heart rate contamination:** Identify and isolate baseline maternal HR artefacts
- **Normalisation:** Per-recording z-score scaling to handle inter-patient variability

**Deliverable:**
A `preprocessing.py` module with reusable functions:
```python
load_record(record_id) → raw_fhr, raw_uc, metadata
process_record(record_id) → fhr_clean, uc_clean, metadata
```
Output: Preprocessed signals ready for segmentation and feature extraction.

---

### Block C: Segmentation Strategy

**Objective:** Convert 90-minute continuous recordings into fixed-length windows suitable for machine learning.

**Key Design Decisions (You Must Decide):**
1. **Window size:** 
   - 1 minute: More granular, more data points, higher noise
   - 5 minutes: Balanced, captures longer patterns
   - 10 minutes: Coarser, fewer samples
   - **Recommendation:** Start with 5 minutes
2. **Stride (overlap):**
   - Non-overlapping: Fewer samples, faster processing
   - 50% overlap: More samples, captures transitions between windows
   - **Recommendation:** Start non-overlapping; try overlap if data is limited
3. **Label assignment (critical for weakly-supervised learning):**
   - All segments from a "normal" recording → label = 0
   - All segments from a "pathological" recording → label = 1
   - This introduces **label noise** (some "normal" segments may have transient fetal stress)
   - Document this assumption explicitly in methodology

**Deliverable:**
A `segmentation.py` module that:
- Takes a cleaned recording + global outcome label
- Returns matrix of shape (N_segments, window_length, 2) for FHR + UC
- Logs segmentation statistics (total segments created, distribution)
- Outputs a segmentation manifest CSV: `[segment_id, source_record, window_start_s, window_end_s, label]`

---

### Block D: Feature Engineering Toolkit

**Objective:** Encode clinical and physiological knowledge into interpretable features before training Random Forest.

**Feature Categories to Extract:**

| **Category** | **Examples** | **Clinical Relevance** |
|---|---|---|
| **Baseline** | Mean FHR, Median FHR | Reflects resting fetal heart rate |
| **Variability (Time-Domain)** | Short-term variability, Long-term variability, Standard deviation, Range | Indicates fetal nervous system integrity |
| **Decelerations** | Count, Mean depth, Mean duration, Nadir recovery slope | Key FIGO feature for abnormality |
| **Accelerations** | Count, Mean amplitude, Mean rise time | Sign of fetal responsiveness |
| **Frequency-Domain** | Peak frequencies, Spectral power in VLF/LF/HF bands | Identifies rhythm complexity |
| **Entropy & Complexity** | Sample entropy, ApEn, Lempel-Ziv complexity | Detects loss of fetal adaptive capacity |
| **Uterine Contraction Features** | Mean UC amplitude, UC frequency, Baseline tonus | Labour intensity and patterns |

**Deliverable:**
A `feature_extraction.py` module:
```python
extract_features(fhr_segment, uc_segment) → feature_vector
compute_all_features(segmented_data) → feature_matrix (N × F)
```
Output: Feature matrix (N_segments × N_features) with column names for interpretability.

---

### Block E: Baseline Model – Random Forest

**Objective:** Build a strong, explainable baseline classifier. RF handles noisy labels well and provides feature importance.

**What to Build:**
- **Train/test split:**
  - 70% train / 30% test (stratified by label to preserve class distribution)
  - Consider blocked cross-validation by maternal record (to avoid data leakage)
- **Model configuration:**
  - Random Forest with 100–200 trees
  - Set `max_depth=10` to prevent overfitting
  - Enable `class_weight='balanced'` to handle class imbalance
- **Evaluation metrics:**
  - **ROC-AUC:** Overall discrimination ability
  - **Sensitivity (Recall):** Crucial clinically — catch true pathological cases
  - **Specificity:** Avoid false alarms (normal cases predicted as abnormal)
  - **Confusion matrix:** True positives, false positives, true negatives, false negatives
  - **Feature importance ranking:** Which features drive predictions?

**Deliverable:**
- Trained Random Forest model (pickled)
- Performance report with all metrics
- Feature importance bar chart
- ROC curve plot
- Confusion matrix visualization

---

### Block F: Post-Processing & Temporal Smoothing

**Objective:** Bridge segment-level predictions to clinician-friendly, real-time alerts.

**What You Need:**
- **Aggregation:** Combine segment predictions to recording-level risk score
  - Simple: majority vote over all segments
  - Weighted: give more weight to late-labour segments (closer to delivery)
- **Temporal smoothing:**
  - Moving average filter over time
  - Majority voting (e.g., flag alert if 2 of 3 consecutive segments are abnormal)
  - Prevents flickering between normal/abnormal states
- **Alert thresholding:**
  - Define risk levels: Normal (0), Caution (1), High Risk (2)
  - Set thresholds based on RF probability output
  - Example: P(abnormal) < 0.33 → Normal; 0.33–0.66 → Caution; > 0.66 → High Risk

**Deliverable:**
An `alert_generator.py` module that:
- Takes segment-level predictions + timestamps
- Outputs smoothed, time-stamped alerts
- Simulates real-time streaming scenario
- Produces alert timeline visualization

---

### Block G: Advanced Model – CNN (Optional but Strong Extension)

**Objective:** Compare learned feature representations (CNN) vs. hand-engineered features (RF).

**Only Proceed After Block E (RF baseline works)**

**What to Build:**
- **Architecture:** 1D CNN on raw or minimally-processed FHR + UC time-series
  - Avoids hand-engineered feature bottleneck
  - Learns what matters directly from signals
- **Comparison:**
  - RF AUC vs. CNN AUC
  - Inference time (clinical real-time requirement)
  - Interpretability trade-off
  - ROC curves side-by-side
- **Feature visualization:** Which time intervals / signal patterns does CNN focus on?

**Deliverable:**
- Trained CNN model
- Side-by-side performance comparison (RF vs. CNN)
- Summary of trade-offs (accuracy vs. interpretability vs. speed)

---

### Block H: Evaluation & Limitations

**Objective:** Explicitly document assumptions, biases, and clinical gaps.

**Document Thoroughly:**
- ✅ **Weak label assumption:** Outcome labels are recording-level, not time-localised. Some "normal" segments may contain transient stress.
- ✅ **Dataset bias:** 
  - Czech cohort only (may not generalize globally)
  - Specific delivery protocols at UHB (not universal)
  - Selected from 9164 recordings (potential selection bias)
- ✅ **Class imbalance:** Majority of cases are normal; pathological cases are minority
- ✅ **Inter-observer variability:** CTG interpretation varies among clinicians (FIGO guidelines exist but are subjective)
- ✅ **Need for prospective validation:** This model must be tested on new, prospectively-collected data
- ✅ **Clinical implementation gaps:**
  - Model provides decision support, not diagnosis
  - Human clinician retains final decision authority
  - Real-time system requires <1s inference time

**Deliverable:**
A comprehensive limitations & future work section in final report.

---

### Block I: Real-Time Decision Support Integration (Optional)

**Objective:** Demonstrate the pathway from offline model → AI4Life decision support system.

**What to Show:**
- Simulate real-time CTG streaming using sliding windows
- Output:
  - Real-time risk score
  - Alert levels over time
  - Clinician summary dashboard (visualization)
- Emphasise: Decision support, not autonomous diagnosis

---

## Original Step-by-Step Overview

### Step 1: Understand the Data and Clinical Labels

**Objective:** Know exactly what information you do and do not have.

**Tasks:**
- Explore the PhysioNet CTU-UHB CTG dataset
- Identify:
  - Available signals (FHR, uterine contractions, maternal HR)
  - Sampling rate and duration
  - Outcome labels (e.g. umbilical artery pH, Apgar score, normal/pathological)
- Accept early that:
  - Labels are outcome-based, not time-localised
  - You are doing weakly supervised learning

📌 **This justifies your modelling choices in the report.**

### Step 2: Signal Preprocessing

**Objective:** Clean the data so ML isn't learning artefacts.

**Tasks:**
- Remove or correct:
  - Signal dropouts
  - Spikes and noise
  - Maternal heart rate contamination
- Apply:
  - Interpolation for short gaps
  - Smoothing / median filtering
  - Normalisation (per recording)

**Deliverable:** Cleaned, standardised CTG signals ready for feature extraction

### Step 3: Segmentation (Critical Step)

**Objective:** Turn long recordings into learnable units.

Because there are no timestamps for "things going wrong":
- Segment signals into fixed windows (e.g. 1–5 minutes)
- Assign each segment the global outcome label (normal vs pathological)
- This introduces label noise — and that's OK

**Make sure you explicitly state this in your methodology.**

### Step 4: Feature Engineering (Before Any Neural Networks)

**Objective:** Encode clinical and physiological knowledge.

**Extract features such as:**
- **Baseline heart rate**
- **Short-term variability**
- **Long-term variability**
- **Decelerations**
  - Depth
  - Duration
  - Frequency
- **Acceleration statistics**
- **Frequency-domain features**
  - Power spectral density
  - Entropy / complexity measures

📌 **This step makes your work:**
- Interpretable
- Clinically meaningful
- Much easier to defend in viva / review

### Step 5: First Model — Random Forest (Excellent Choice)

**Objective:** Build a strong, explainable baseline.

**Why Random Forest first:**
- Handles noisy labels well
- Works with small-to-medium datasets
- Gives feature importance
- Easy to debug

**Tasks:**
- Train RF classifier on window-level features
- Evaluate using:
  - ROC-AUC
  - Sensitivity (very important clinically)
  - Specificity
- Analyse:
  - Which features drive predictions
  - Failure cases

**This step validates whether the signal actually contains predictive information.**

### Step 6: Post-Processing for Temporal Consistency

**Objective:** Avoid noisy, jumpy predictions.

Since clinicians don't want flickering alerts:
- Smooth predictions over time
- Use majority voting or moving averages
- Trigger alerts only when risk is sustained

📌 **This bridges ML output → clinical usability.**

### Step 7: Advanced Model — CNN (Optional but Strong Extension)

**Objective:** Learn features directly from raw signals.

Only do this after the RF works.

**Approach:**
- Use 1D CNNs on segmented signals
- Compare:
  - CNN vs Random Forest
  - Performance vs interpretability
  - Highlight trade-offs in the report

**This shows methodological maturity, not "NN for the sake of it".**

### Step 8: Real-Time Decision Support Concept

**Objective:** Tie everything back to AI4Life.

- Simulate real-time streaming using sliding windows
- **Output:**
  - Risk score
  - Alert levels (normal / caution / high risk)
- **Emphasise:**
  - Decision support, not diagnosis
  - Human-in-the-loop design

### Step 9: Evaluation and Limitations

Be explicit about:
- Weak labels
- Inter-observer variability in CTG interpretation
- Dataset bias
- Need for prospective clinical validation
