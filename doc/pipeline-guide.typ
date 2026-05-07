= CTG Classifier Pipeline Guide

This document summarizes the practical foundation for building a machine learning classifier for cardiotocography (CTG) analysis. It is written as a lightweight implementation guide aligned with the current project pipeline.

== Objective

Build a reproducible classifier that maps CTG signal windows to clinical risk classes (for example: Normal, Suspicious, Pathological) while balancing:

- diagnostic performance (especially minority-risk recall),
- robustness to noisy intrapartum signals,
- interpretability for clinician trust,
- and deployment feasibility (speed, size, maintainability).

== End-to-End Process

The workflow follows nine stages:

1. Define prediction target and label policy.
2. Assemble and audit dataset quality.
3. Preprocess FHR and UC signals.
4. Segment recordings into fixed windows.
5. Extract clinically grounded features (for tabular models).
6. Train models with leakage-aware grouped splits.
7. Evaluate with class-aware and stability metrics.
8. Perform interpretability and error analysis.
9. Package artefacts for reproducibility and future extension.

== 1) Problem and Label Definition

- Select the task type: binary risk detection or multi-class risk stratification.
- Define how labels are generated (event-level annotation vs record-level weak labels).
- Fix class semantics early (e.g., what qualifies as Pathological).
- Predefine the primary safety metric (typically pathological recall or macro-F1).

Why this matters: weak or inconsistent labels usually cap model performance more than algorithm choice.

== 2) Dataset and Governance Foundation

- Use a documented cohort (for example CTU-UHB) and track inclusion/exclusion criteria.
- Keep record identifiers intact for grouped splitting.
- Report class distribution at record and window levels.
- Store data provenance, preprocessing versions, and split manifests.

Minimum data checks:

- signal duration coverage,
- missingness and dropout rates,
- outlier/spike prevalence,
- and label completeness.

== 3) Signal Preprocessing

Core preprocessing for fetal heart rate (FHR) and uterine contraction (UC):

- physiological range checks,
- spike/outlier filtering,
- short-gap interpolation (bounded),
- long-gap exclusion,
- smoothing and normalization,
- quality scoring per window.

Design principle: preserve physiological structure while removing non-clinical artefacts.

== 4) Windowing Strategy

- Segment each recording into fixed-length windows.
- Attach weak labels consistently from the chosen policy.
- Reject windows that fail quality thresholds (valid sample ratio, continuity).
- Keep metadata (record id, timestamp, quality stats) for auditing and grouped evaluation.

Important: split by record, not by window, to avoid leakage.

== 5) Feature Engineering Foundation

For feature-based models (e.g., Random Forest), include domains that map to CTG physiology:

- baseline descriptors,
- variability metrics,
- acceleration/deceleration burden,
- UC morphology and timing,
- spectral and entropy measures,
- quality indicators.

Use fixed extraction rules and units. Version feature schemas so results are reproducible across reruns.

== 6) Model Families and Training

A practical baseline pairing:

- Feature-based tabular model (Random Forest): strong baseline, efficient, easier to interpret.
- Raw-signal deep model (1D-CNN): tests representation learning from waveform morphology.

Training rules:

- grouped train/validation/test split by record id,
- class-weighted loss or cost-sensitive learning,
- identical preprocessing across model families,
- fixed random seeds and tracked hyperparameter budgets.

== 7) Evaluation Framework

Do not rely on accuracy alone. Report:

- macro-F1,
- weighted-F1,
- balanced accuracy,
- class-wise precision/recall/specificity,
- confusion matrix,
- and ROC-AUC (if probability outputs are stable).

Add operational metrics:

- training time,
- inference latency,
- model size.

Add robustness metrics:

- repeated grouped runs with mean/std,
- held-out grouped test performance gap analysis.

== 8) Interpretability and Clinical Trust

- Use global importance (e.g., feature importance, SHAP) for cohort-level behavior.
- Use local explanations for case-level review where needed.
- Cross-check explanations against known CTG constructs (baseline shifts, reduced variability, deceleration burden, contraction timing).
- Pair explanation outputs with error analysis, especially Suspicious vs Pathological confusion.

Goal: transparent support for clinical reasoning, not black-box scoring.

== 9) Reproducibility Package

At minimum, preserve:

- preprocessing parameters,
- split manifests,
- model configs and seeds,
- metric reports and confusion matrices,
- feature/importance exports,
- and run logs.

This turns a one-off experiment into a benchmarkable, extensible pipeline.

== Practical Baseline Recommendation

For the current project stage, start with a leakage-aware feature pipeline + Random Forest baseline, then compare against a 1D-CNN under identical grouped splits. This gives a defensible benchmark on:

- performance,
- stability,
- interpretability,
- and runtime trade-offs.

== Common Failure Points to Avoid

- Random window-level splits (data leakage).
- Inconsistent preprocessing between models.
- Accuracy-only reporting with imbalanced classes.
- Ignoring label uncertainty from weak supervision.
- No split-seed sensitivity analysis.
- Missing audit trail for reproducibility.

== Extension Path

After the baseline is stable, extend in this order:

1. External validation on a second cohort.
2. Threshold calibration for clinical alert balance.
3. Ensemble or hybrid methods under the same protocol.
4. Human-in-the-loop evaluation with clinician feedback.

This progression keeps the project scientifically rigorous while moving toward deployable CTG decision support.
