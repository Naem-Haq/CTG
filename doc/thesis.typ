// FYP thesis document (Typst)
#import "graduate-thesis.typ": graduate-thesis

#show: graduate-thesis.with(
  title: [Benchmarking Machine Learning Classifiers for Fetal Heart Rate Pattern Detection in Cardiotocography],
  author: "Naem Haq (23379243)",
  degree: [Bachelor/Master of Science in Immersive Software Engineering],
  department: [Department of Computer Science and Information Systems],
  university: [University of Limerick],
  supervisor: [Dr. Salaheddin Alakkari],
  month: [May],
  year: [2026],
  degree-year: [2026],
  program-type: [Bachelor of Science],
  degree-type: [Bachelor],
  degree-department: [Department of Computer Science and Information Systems],
  location: [Limerick, Ireland],
  word-count: [9,520],
  word-count-with-citations: [9,964],
  abstract: [
    Cardiotocography (CTG) remains the primary clinical tool for monitoring fetal wellbeing during labour by recording fetal heart rate (FHR) and uterine contractions. However, interpretation of CTG traces is often subjective, leading to inconsistent clinical decisions and variable outcomes. This dissertation applies biomedical signal processing and supervised machine learning to support detection and classification of clinically relevant fetal heart rate patterns, including deceleration-linked risk states.

    The completed benchmark implements a reproducible pipeline covering signal quality control, preprocessing, leakage-aware grouped splitting, feature extraction, and comparative modelling with Random Forest (RF) and a 1D convolutional neural network (1D-CNN). Evaluation reports class-wise and aggregate performance, efficiency evidence, and interpretable diagnostics using confusion analysis, threshold trade-off behaviour, and feature/domain-importance outputs.

    Results show that the RF baseline outperforms the tested CNN across headline discrimination metrics and provides more stable class behaviour under this data regime. Findings are intentionally bounded to within-cohort benchmarking on CTU-UHB with weakly supervised labels, while the dissertation provides a transparent, auditable baseline for future external validation and prospective clinical translation.
  ],
  keywords: [Cardiotocography; Fetal monitoring; Machine learning; Random forest; Convolutional neural network; Clinical decision support],
  acknowledgments: [
    I would like to express my sincere gratitude to my supervisor, Dr. Salaheddin Alakkari, for his continuous guidance, insightful feedback, and support throughout the development of this project. I am also thankful to the INFANT Research Centre and the School of Engineering at University College Cork for providing the resources and expertise that have made this research possible.

    My appreciation extends to the clinicians and researchers involved in the AI4Life project, whose work and discussions have inspired many aspects of this study. Finally, I would like to thank my family and peers for their encouragement and support during this stage of my academic journey.
  ],
  acronyms: (
    "CTG": "Cardiotocography",
    "CTU-UHB": "CTU-UHB Intrapartum Cardiotocography Database",
    "FHR": "Fetal Heart Rate",
    "UC": "Uterine Contractions",
    "MHR": "Maternal Heart Rate",
    "ECG": "Electrocardiogram",
    "FIGO": "International Federation of Gynecology and Obstetrics",
    "RF": "Random Forest",
    "CNN": "Convolutional Neural Network",
    "SVM": "Support Vector Machine",
    "AUC": "Area Under the ROC Curve",
    "ROC": "Receiver Operating Characteristic",
    "SHAP": "SHapley Additive exPlanations",
    "LIME": "Local Interpretable Model-agnostic Explanations",
    "XAI": "Explainable Artificial Intelligence",
    "ML": "Machine Learning",
    "AI": "Artificial Intelligence",
    "JSON": "JavaScript Object Notation",
    "CSV": "Comma-Separated Values",
    "CPU": "Central Processing Unit",
    "RAM": "Random Access Memory",
  ),
)

= Introduction

== Background and Context

Cardiotocography (CTG) is a core technology in intrapartum care, combining continuous fetal heart rate and uterine activity monitoring to support decisions during labour. International guidance positions CTG as a key component of fetal surveillance, especially when labour is clinically complex or risk factors are present #cite(<figo2015_intro>). However, the physiological meaning of CTG patterns is often context-dependent, and visual interpretation can vary between clinicians. Recent evidence syntheses show that this subjectivity remains a persistent challenge, with uneven performance across studies, settings, and outcome definitions #cite(<francis2024_scoping>).

These limitations have motivated a shift toward computational analysis of CTG signals. Traditional machine learning methods such as support vector machines and random forests have demonstrated promising classification performance on curated datasets #cite(<nagendra2017_realtime>); #cite(<hoodbhoy2019_ml>); #cite(<khare2022_compare>). More recent deep learning and ensemble approaches continue this trend, but also highlight issues of dataset shift, benchmarking consistency, and external validity #cite(<zhang2024_svmcnn>); #cite(<chiou2025_npj>); #cite(<mendis2025_crossdb>). In parallel, explainability has become central to clinical adoption, with SHAP- and LIME-based methods used to expose feature influence and improve trust in model outputs #cite(<feng2023_shap>); #cite(<ribeiro2016_lime>). This dissertation addresses that context with a completed, evidence-backed benchmark designed for reproducibility and clinically interpretable comparison. The benchmarking focus was motivated during the author's Residency 4 placement at the INFANT Research Centre, where CTG interpretation variability highlighted the need for a leakage-aware evaluation framework.

== Problem Statement

Despite growing interest in machine learning for CTG, there is still no clear consensus on which modelling approach is most reliable for intrapartum use because studies vary widely in datasets, preprocessing, labels, and reporting standards #cite(<khare2022_compare>); #cite(<francis2024_scoping>). This dissertation addresses that gap by benchmarking supervised classifiers for fetal heart rate pattern detection under standardized, leakage-aware experimental conditions. The core problem is to improve detection of clinically important compromise signals, especially deceleration-related risk patterns, while handling noisy CTG signals, class imbalance, and conflicting sensitivity-specificity trade-offs #cite(<nazli2025_imbalanced>); #cite(<zhang2024_svmcnn>). It also requires fair comparison of feature-based and deep models under identical split policies and evaluation metrics. The benchmark prioritizes interpretability and cross-dataset robustness so outputs remain clinically understandable and transferable beyond a single cohort #cite(<feng2023_shap>); #cite(<mendis2025_crossdb>).

== Research Objectives

This dissertation delivers a rigorous benchmark of supervised machine learning classifiers for automated fetal heart rate pattern classification from CTG under standardized experimental conditions. The objectives are to implement a reproducible preprocessing pipeline for artifact handling and signal conditioning #cite(<clifford2014_fecg>), engineer clinically grounded time- and frequency-domain features, and train a representative set of classifiers with consistent hyperparameter tuning and grouped validation. It compares models using clinically meaningful metrics and statistical testing, assesses explanation quality using SHAP and local interpretation methods, and provides practical recommendations that balance diagnostic performance, interpretability, and deployment feasibility for intrapartum decision support #cite(<feng2023_shap>); #cite(<ribeiro2016_lime>).

== Research Questions

- RQ1: Which supervised classifier (feature-based Random Forest or 1D-CNN) delivers the strongest CTG classification performance on data using accuracy, recall, specificity, F1, and AUC?
- RQ2: How do preprocessing and engineered feature sets influence discrimination of Normal, Suspicious, and Pathological windows, particularly pathological-class recall?
- RQ3: What efficiency trade-offs exist between models in training time, inference time, and model size for practical decision-support deployment?
- RQ4: How stable are fixed-split results across window-level and record-level evaluation views, and what split-composition diagnostics indicate within-cohort model generalisability?

== Project Scope and Limitations

This project is scoped to benchmarking intrapartum CTG classification using fetal heart rate and uterine contraction signals from the CTU-UHB database, with weakly supervised window labels derived from record-level outcomes #cite(<chudacek2014_open>). It implements a reproducible Python workflow covering preprocessing, artifact handling, feature extraction, grouped splitting, and comparative modelling with Random Forest and 1D-CNN baselines #cite(<clifford2014_fecg>); #cite(<fergus2020_cnn>). Benchmarking reports performance metrics (accuracy, recall, specificity, F1, AUC), efficiency metrics (training time, inference time, model size), and generalisability checks via split-composition diagnostics and held-out grouped testing. Limitations include retrospective design, outcome-label noise, class imbalance, single-cohort dependence, and no prospective deployment or obstetrician-in-the-loop validation. The study also excludes multimodal inputs, including fetal ECG and rich maternal covariates, so external transferability remains to be established. Interpretability is assessed with feature importance and SHAP-style attribution, but human factors evaluation of presentation and workflow integration is outside scope #cite(<francis2024_scoping>); #cite(<mendis2025_crossdb>).

== Expected Contributions

This dissertation contributes a reproducible benchmarking framework for intrapartum CTG classification that standardizes preprocessing, grouped splitting, and metric reporting across models, reducing ambiguity in prior comparisons and clarifying fair comparison standards in CTG #cite(<francis2024_scoping>). It delivers an end-to-end Python pipeline that links signal conditioning, feature engineering, supervised learning, efficiency profiling, and interpretability analysis in one transparent workflow #cite(<clifford2014_fecg>); #cite(<feng2023_shap>). Scientifically, the study provides controlled evidence on performance differences between feature-based and deep approaches, including class-wise behaviour under imbalance and generalisation across validation settings #cite(<zhang2024_svmcnn>); #cite(<mendis2025_crossdb>). Practically, it offers actionable guidance for model selection in clinician-facing decision support by balancing diagnostic accuracy, specificity, runtime constraints, and explanation quality, while establishing a defensible baseline for future multimodal integration and prospective clinical evaluation #cite(<chiou2025_npj>). It also provides traceable artefact packages and reporting templates that strengthen replication, supervisor review, and audit readiness for subsequent cohorts and extensions.

== Dissertation Outline

The remainder of this dissertation is organized as follows:

*Chapter 2:* reviews CTG clinical context, signal processing methods, machine learning approaches, benchmarking principles, and interpretability requirements, and defines the research gap.

*Chapter 3:* details the methodology, including dataset preparation, preprocessing, feature extraction, model design, training strategy, and evaluation framework.

*Chapter 4:* presents results for data quality, classifier performance, efficiency, and generalisability analyses.

*Chapter 5:* interprets findings against prior literature, discusses clinical implications, and examines limitations and validity threats.

*Chapter 6:* summarizes contributions, practical recommendations, and future research priorities.

*Appendices:* provide implementation details, parameter settings, supplementary figures and tables, and reproducibility materials. This structure is intentionally aligned to the benchmarking workflow so readers can trace claims from methods to artefacts without narrative gaps.

= Literature Review and Background

== Cardiotocography: Clinical Context

Cardiotocography (CTG) is the principal intrapartum monitoring modality used to assess fetal wellbeing by recording fetal heart rate alongside uterine activity during labour. Its clinical rationale is physiological: evolving hypoxia and reduced placental reserve can alter autonomic control of heart rate, producing patterns that may precede metabolic compromise. For this reason, CTG is embedded in routine obstetric pathways and is commonly used when labour risk is elevated or continuous surveillance is indicated #cite(<figo2015_intro>). Standard interpretation frameworks evaluate baseline rate, baseline variability, accelerations, and decelerations, then integrate these features with contraction timing and overall trace evolution to support escalation or conservative management decisions #cite(<figo2015_intro>). In practice, deceleration morphology and timing remain especially influential because they are linked to potential cord compression, head compression, or uteroplacental insufficiency, yet these same patterns are often the most contested in real-time interpretation. Contemporary evidence indicates that CTG remains clinically indispensable but limited by variable observer agreement and inconsistent translation of patterns into outcomes, motivating decision support and improved analytic methods #cite(<francis2024_scoping>). Large databases have further exposed heterogeneity in trace quality, labeling, and outcomes, reinforcing the need for interpretation #cite(<chudacek2014_open>). Consequently, understanding CTG’s clinical context requires both physiological interpretation and awareness of uncertainty, since safe use depends not only on signal patterns, but on reproducible assessment frameworks and response during labour.
This interpretive variability also complicates benchmarking, because outcome labels can encode local practice norms rather than fully transferable physiologic ground truth.

== Signal Processing for CTG Analysis

Signal processing is a foundational stage in automated CTG analysis because raw fetal heart rate (FHR) and uterine contraction traces contain dropout, spikes, and non-physiological fluctuations that can mislead downstream models. A robust pipeline therefore begins with quality control, physiologic range checks, short-gap interpolation, smoothing, and normalization, so extracted patterns reflect fetal physiology rather than sensor artefact #cite(<clifford2014_fecg>); #cite(<spilka2013_autoeval>). Baseline estimation is then required to contextualize transient events; moving-window and median-based approaches are commonly used because they suppress short oscillations while preserving longer trends relevant to clinical interpretation #cite(<figo2015_intro>). After baseline stabilisation, acceleration and deceleration candidates can be identified from duration- and amplitude-constrained deviations, often refined by temporal alignment with contraction activity to improve event characterization. Where missingness is prolonged, affected windows are excluded to prevent interpolation artefacts dominating learned patterns.

Feature extraction translates cleaned signals into model-ready representations. In this project, features are designed to capture complementary information across time-domain, variability, morphology, and frequency-oriented descriptors, including baseline level, short- and long-term variability surrogates, deceleration burden, and contraction-linked dynamics. This design supports both discrimination and interpretability, allowing model outputs to be traced back to clinically meaningful behaviour. Prior CTG studies consistently show that preprocessing quality and feature construction materially influence classification performance and reproducibility, often more than algorithm choice alone #cite(<hoodbhoy2019_ml>); #cite(<francis2024_scoping>). For this reason, signal processing is treated not as a preliminary step, but as a primary methodological component of trustworthy CTG benchmarking.
Critically, weak preprocessing can imitate pathological variability and cause models to learn artefact signatures, inflating apparent gains that fail under cleaner evaluation.

== Machine Learning in Fetal Monitoring

Machine learning has become central to automated fetal monitoring because CTG interpretation involves complex, noisy patterns that are difficult to assess consistently at the bedside. Most published systems use supervised learning, where models are trained on labelled CTG segments or recordings to classify fetal condition and support escalation decisions #cite(<hoodbhoy2019_ml>); #cite(<francis2024_scoping>). In this literature, traditional tabular models remain highly competitive when feature engineering is clinically informed. Support Vector Machines, Random Forests, gradient boosting methods, and related classifiers are frequently reported to achieve strong discrimination on curated datasets, particularly when preprocessing quality, class balancing, and leakage control are handled carefully #cite(<nagendra2017_realtime>); #cite(<khare2022_compare>); #cite(<innab2024_lgbm>). These models are attractive in clinical contexts because they are comparatively efficient and easier to interpret through feature importance analyses.

Deep learning has expanded CTG research by learning representations directly from raw or minimally processed time series. One-dimensional convolutional networks are widely studied for this purpose, and recent work shows they can capture local waveform structure associated with fetal compromise, especially in larger datasets #cite(<cao2023_fusion>); #cite(<zhang2024_svmcnn>); #cite(<chiou2025_npj>). However, deep models are more sensitive to label quality, data imbalance, and distribution shift, and reported performance varies substantially across cohorts and evaluation protocols. Cross-database experiments highlight that gains observed on single-center benchmarks may not transfer reliably without stronger validation design and calibration strategies #cite(<mendis2025_crossdb>).

Ensemble and hybrid approaches attempt to combine strengths of multiple learners, often improving robustness by reducing variance and exploiting complementary decision boundaries. Recent CTG studies using stacked or threshold-optimized ensembles report improved macro-level performance, but benefits depend on transparent benchmarking and clinically meaningful error analysis rather than headline accuracy alone #cite(<kong2025_ensemble>); #cite(<nazli2025_imbalanced>). Consequently, current best practice emphasizes three requirements: rigorous split strategy and metric reporting, explicit handling of imbalance and temporal context, and interpretability methods such as SHAP or local explanations to support clinician trust #cite(<feng2023_shap>); #cite(<ribeiro2016_lime>). This evidence base motivates comparative evaluation of interpretable and deep models within a reproducible framework for fetal monitoring decision support. Accordingly, this dissertation benchmarks feature-based and raw-signal pipelines, contrasts performance with efficiency, and reports fixed-split, grouped held-out evidence across window-level and record-level views to prioritize clinically deployable rather than retrospective model success.
However, many published gains remain difficult to interrogate because preprocessing detail, tuning budget parity, and leakage controls are incompletely reported. Studies that emphasise headline accuracy without class-wise failure pathways, calibration analysis, or external checks cannot reliably separate physiologic learning from cohort-specific shortcut learning in practice.

== Benchmarking and Performance Evaluation

Benchmarking is essential in CTG machine learning because reported performance is highly sensitive to dataset composition, label design, preprocessing, and split strategy. In fetal monitoring, class imbalance is common, so single-metric reporting can be misleading: high overall accuracy may coexist with poor detection of minority pathological cases. Accordingly, evaluation should include class-aware metrics such as recall, specificity, precision, macro-F1, and balanced accuracy, alongside threshold-independent summaries such as AUC, to reflect clinically relevant trade-offs between missed compromise and unnecessary escalation #cite(<francis2024_scoping>); #cite(<nazli2025_imbalanced>).

For fair model comparison, benchmarking protocols must control data leakage and maintain consistent experimental conditions across classifiers. In CTG, grouped splitting at recording level is particularly important because adjacent windows from the same trace are strongly correlated; random window-level splitting can therefore inflate performance estimates. Recent comparative studies also show that conclusions vary when metric definitions, class weighting, and hyperparameter search budgets differ, reinforcing the need for transparent protocols and reproducible reporting standards #cite(<khare2022_compare>); #cite(<kong2025_ensemble>); #cite(<m2025_icitiit>).

Benchmarking guidance in the literature also recommends extending performance comparison beyond discrimination metrics to include efficiency and robustness. Training time, inference latency, and model size influence practical deployment in decision-support settings, especially when near-real-time alerting is required. Generalisability is often examined through repeated grouped cross-validation, mean and standard deviation reporting, and independent held-out testing. Where possible, cross-database evaluation provides a stronger estimate of external validity and exposes distribution shift effects that are not visible in single-cohort experiments #cite(<mendis2025_crossdb>).

Finally, benchmarking guidance indicates that statistical comparison helps distinguish true model differences from random variation. Confidence intervals and paired significance testing across folds or test units strengthen evidential claims and support defensible model selection for clinical translation #cite(<francis2024_scoping>). In this dissertation, these elements are treated as reference standards and as extensions for future repeated-resampling work, rather than as completed inferential outputs from the fixed-split benchmark. These safeguards improve comparability, reproducibility, and clinical trust in CTG AI evidence.

== Trustworthy AI and Clinical Interpretability

Trustworthy AI is a prerequisite for clinical adoption of automated CTG interpretation, because predictions influence high-stakes decisions during labour. In this context, trustworthiness extends beyond discrimination performance to include transparency, robustness, fairness awareness, and accountability. Recent CTG literature shows that models can achieve strong classification metrics, yet deployment confidence remains limited when decision logic is opaque or unstable across datasets #cite(<francis2024_scoping>); #cite(<mendis2025_crossdb>).

Interpretability methods are therefore central. Global explanation tools, including feature importance and SHAP-based attribution, help clinicians understand which signal characteristics drive model behaviour at cohort level, while local methods such as LIME and instance-level SHAP clarify why a specific prediction was generated for a particular window or recording #cite(<feng2023_shap>); #cite(<ribeiro2016_lime>). For CTG, interpretable outputs are most useful when they map to clinically meaningful constructs, such as baseline shifts, variability reduction, deceleration burden, and contraction-related timing patterns.

A practical challenge is the interpretability-performance trade-off. Highly expressive deep or ensemble models may improve accuracy but can reduce auditability, whereas simpler models improve traceability but may underfit complex physiology. Current best practice is not to choose one objective over the other, but to benchmark both and report performance, efficiency, and explanation quality together #cite(<chiou2025_npj>); #cite(<kong2025_ensemble>). Clinically, trustworthy deployment also requires human-in-the-loop governance, clear threshold policies, and monitoring for dataset shift, so decision support augments rather than replaces obstetric judgment during intrapartum care.
Critically, explanation quality is often asserted rather than evaluated: many studies show attribution plots but do not test clinical agreement, stability, or bedside actionability.

== Related Work and Research Gap

Research on automated CTG interpretation has expanded rapidly, with studies progressing from handcrafted feature models to deep representation learning. Earlier work commonly used supervised classifiers such as Support Vector Machines, Random Forests, and related tabular methods trained on engineered fetal heart rate and uterine activity features, often reporting strong performance on curated datasets #cite(<nagendra2017_realtime>); #cite(<hoodbhoy2019_ml>); #cite(<khare2022_compare>). More recent studies have introduced boosting, hybrid pipelines, and deep networks, including 1D-CNN architectures that learn directly from waveform segments and can capture local temporal morphology associated with fetal compromise #cite(<cao2023_fusion>); #cite(<zhang2024_svmcnn>); #cite(<chiou2025_npj>). Ensemble approaches have further improved aggregate metrics in some settings by combining complementary learners #cite(<kong2025_ensemble>).

Despite these advances, the literature remains difficult to compare directly. Reviews consistently report heterogeneity in dataset choice, label definitions, preprocessing assumptions, class handling, and reporting practice, which limits reproducibility and weakens conclusions about model superiority #cite(<francis2024_scoping>). A recurring concern is optimistic estimation caused by split design: window-level random splits can leak subject-specific patterns across train and test sets, inflating apparent performance. External validity is another persistent gap, as cross-database evaluations frequently show reduced transfer performance relative to single-cohort results #cite(<mendis2025_crossdb>). In parallel, many papers prioritize discrimination metrics without equivalent attention to efficiency constraints or deployment practicality.

Interpretability is also unevenly addressed. While explainability tools such as SHAP and local post-hoc methods are increasingly used, many studies still provide limited linkage between model attributions and clinically meaningful CTG constructs #cite(<feng2023_shap>); #cite(<ribeiro2016_lime>).

Evidence strength therefore varies materially: studies with grouped leakage control, explicit class-aware reporting, label definitions, and external checks provide higher-confidence conclusions than single-split reports with sparse protocol detail. This gap motivates treating benchmarking design quality as a first-order comparison variable rather than a secondary implementation detail.

This dissertation addresses these gaps through a controlled benchmarking framework built on reproducible preprocessing, grouped leakage-aware splitting, and unified evaluation of performance, efficiency, and generalisability. It compares feature-based and raw-signal modelling under the same protocol, reports class-wise and aggregate metrics with split-composition diagnostics and held-out comparison, and integrates interpretability analysis anchored to clinically relevant signal behaviour. The objective is not only higher scores, but defensible evidence for practical, clinician-facing CTG decision support within routine intrapartum care.

= Methodology

== Research Design

This study adopts a quantitative experimental benchmarking design to compare CTG classifiers under controlled and reproducible conditions. The design aligns preprocessing, feature construction, split policy, and metric reporting so that observed differences reflect model behaviour rather than pipeline variation #cite(<francis2024_scoping>); #cite(<khare2022_compare>). Data are segmented into weakly labelled windows and evaluated with record-level grouped partitioning to prevent leakage between training and test sets, an essential control for correlated intrapartum traces #cite(<chudacek2014_open>); #cite(<spilka2013_autoeval>). Two modelling families are assessed under a common protocol: feature-based Random Forest and raw-signal 1D-CNN, enabling comparison of interpretability and representation-learning capacity #cite(<fergus2020_cnn>); #cite(<chiou2025_npj>). The executed protocol for this dissertation is a fixed grouped holdout benchmark with class-wise and aggregate discrimination metrics, efficiency evidence, and post-hoc interpretability outputs. Repeated grouped resampling and formal paired significance testing are defined as future extensions, not as completed analyses in this final report. All reported tables, figures, and model artefacts are generated from this fixed protocol to preserve traceability between methods and conclusions. This explicit scope keeps claims aligned with delivered artefacts while preserving reproducibility and auditability.

== Datasets

This project uses the PhysioNet CTU-UHB Intrapartum Cardiotocography Database as the sole modelling dataset for the current dissertation stage. CTU-UHB contains 552 intrapartum recordings collected at University Hospital Brno, with fetal heart rate and uterine contraction signals sampled at 4 Hz and linked to delivery outcomes #cite(<chudacek2014_open>); #cite(<spilka2013_autoeval>). The dataset is widely used in CTG research, which improves comparability with prior benchmarks and supports transparent replication across studies. Labels are derived from post-delivery clinical outcomes rather than event-level timestamps, so segmented windows are treated as weakly supervised samples and evaluated with leakage-aware grouping by recording.

A Cork University Maternity Hospital dataset is part of the broader project context, but it is not included in this FYP benchmark because ethics and governance approvals for modelling use are outside the current project timeline. Consequently, all reported experiments, metrics, and model comparisons are based on CTU-UHB only. Inclusion criteria require sufficient recording duration, usable signal quality after preprocessing, and available outcome variables for class assignment; records with severe dropout, irrecoverable artefact contamination, or missing core labels are excluded. No synthetic augmentation is used at dataset assembly stage. Grouped train-validation-test partitioning is then applied to preserve clinically realistic class structure and prevent patient-level leakage, enabling fair assessment of performance, efficiency, and generalisability within a reproducible experimental protocol. Inclusion and exclusion decisions are retained in exported manifests for auditability.

== Signal Preprocessing Pipeline

The signal preprocessing pipeline is designed to transform raw intrapartum CTG recordings into stable, comparable windows for downstream modelling while preserving clinically meaningful structure. Raw fetal heart rate and uterine contraction signals first undergo physiologic plausibility checks to detect values outside expected bounds, abrupt single-sample spikes, and flat segments suggestive of sensor dropout rather than fetal physiology #cite(<clifford2014_fecg>); #cite(<spilka2013_autoeval>). Quality control flags are generated at sample and window level so low-quality regions can be tracked transparently through later stages. Short gaps are repaired using linear interpolation under a fixed maximum duration rule, whereas longer missing segments are treated as non-recoverable and excluded from model windows to avoid introducing synthetic dynamics. After gap handling, smoothing is applied to reduce high-frequency noise and improve baseline stability before event and variability feature computation. Window acceptance requires minimum valid-sample proportion and continuity criteria, ensuring extracted features represent sustained physiology rather than short artefactual bursts that could distort clinical class boundaries in imbalanced learning settings.

The cleaned streams are then segmented into fixed-length windows aligned to the benchmarking protocol, with each window inheriting weak labels derived from record-level outcomes. To reduce non-clinical amplitude variation across recordings, per-window normalization is applied after filtering and interpolation. Potential maternal heart rate contamination is not modelled with a separate source-separation module in this implementation; instead, its impact is mitigated through plausibility constraints, continuity checks, and exclusion of ambiguous segments. In addition, preprocessing outputs include window quality metadata, dropout duration statistics, and trace-level retention indicators used during split auditing and error analysis. This allows failed or borderline windows to be inspected alongside model predictions, improving diagnostic transparency during development. All preprocessing parameters are fixed, versioned, and reused identically across model families to ensure fair comparison and prevent pipeline-induced performance inflation #cite(<francis2024_scoping>); #cite(<mendis2025_crossdb>). This design supports leakage-aware benchmarking, reproducibility, and interpretable feature extraction grounded in obstetric signal characteristics #cite(<figo2015_intro>); #cite(<chudacek2014_open>). Detailed thresholds, quality rules, and exclusion counts are documented in Appendix B for full auditability and replication.

== Feature Extraction

Feature extraction converts cleaned CTG windows into quantitative descriptors used by downstream classifiers. In this implementation, features are computed per window from both fetal heart rate and uterine contraction channels after preprocessing, with extraction logic fixed across all models to preserve benchmark fairness. The baseline fetal heart rate is estimated using a robust median, then used as the reference for acceleration and deceleration detection with clinically familiar thresholds (15 bpm amplitude, minimum 15 seconds duration) #cite(<figo2015_intro>). For each event type, the pipeline records count, total duration, mean and maximum duration, amplitude summaries, and area under deviation, capturing both frequency and severity of transient changes.

Beyond event features, time-domain descriptors include central tendency and variability measures such as mean, median, trimmed mean, standard deviation, median absolute deviation, interquartile range, and mean absolute first difference. A long-term variability proxy is calculated as the standard deviation of one-minute mean FHR segments. Frequency and complexity information is added through Welch bandpower in low and higher bands, spectral entropy, permutation entropy, and sample entropy estimated after downsampling for computational efficiency #cite(<hoodbhoy2019_ml>); #cite(<francis2024_scoping>). Uterine contraction features include distribution statistics, peak count, peak rate, prominence, inter-peak interval, and contraction AUC derived from peak-based morphology analysis #cite(<spilka2013_autoeval>).

Quality-aware features (post-processing valid percentage and remaining missingness) are retained to contextualize uncertainty. Peak detection uses a minimum inter-peak distance of 75 seconds and adaptive prominence proportional to signal standard deviation, reducing false contraction detections in noisy traces without manual tuning. The final matrix contains 35 leakage-aware model features and is documented in Appendix C with variable names, definitions, thresholds, and units for reproducible reuse.

== Machine Learning Classifiers

Classifier selection in this project is purposefully tied to the benchmarking objective: compare an interpretable, feature-based baseline against a representation-learning model under identical data and split conditions. Rather than optimizing many unrelated algorithms, the design focuses on two model families that reflect a clinically relevant trade-off between transparency and expressive capacity: Random Forest on engineered features, and one-dimensional convolutional neural networks (1D-CNN) on raw FHR+UC windows #cite(<fergus2020_cnn>); #cite(<cao2023_fusion>); #cite(<chiou2025_npj>). This controlled pairing reduces methodological noise and makes performance, efficiency, and generalisability differences attributable to modelling strategy rather than pipeline variation.

The Random Forest classifier is trained on the leakage-aware feature matrix described in Section 3.4. Its strengths are robustness to nonlinearity, tolerance to mixed feature scales, and intrinsic feature importance outputs that support post-hoc clinical interpretation #cite(<hoodbhoy2019_ml>); #cite(<khare2022_compare>). Tree depth, number of estimators, and class weighting are tuned within grouped validation, with emphasis on pathological-class sensitivity and stable macro-level performance. Because RF inference is computationally light and model artefacts are easy to inspect, it serves as the practical reference model for clinician-facing decision support.

The 1D-CNN model is trained directly on fixed-length signal windows to test whether automatic representation learning captures waveform morphology beyond handcrafted descriptors. The architecture uses convolutional feature blocks followed by pooled latent representations and a classification head, optimized with cross-entropy on grouped training splits. This family is expected to model local temporal structure effectively, but is more sensitive to class imbalance, weak labels, and dataset size than tabular baselines #cite(<zhang2024_svmcnn>); #cite(<mendis2025_crossdb>). Accordingly, CNN benchmarking uses the same held-out protocol and shared metric framework as RF, plus efficiency profiling for training/inference cost.

Together, these classifiers operationalize the central methodological question of the dissertation: whether interpretable engineered-feature models remain preferable to deep raw-signal models in this CTG setting. Other classifiers reported in the literature, including SVM, gradient boosting, k-nearest neighbors, and multilayer perceptrons, are treated as contextual comparators rather than primary implementations at this stage. Restricting core experiments prevents fragmented tuning budgets and preserves methodological depth in leakage control, interpretability analysis, and robust error profiling across clinically critical minority classes during evaluation.
This scoped design also simplifies reproducibility checks, since preprocessing, splitting, and metric definitions remain identical across both implemented classifier families.

== Training and Validation Strategy

The training and validation strategy is designed to estimate real-world generalisability while preventing optimistic bias from window-level leakage. Because multiple windows originate from the same labour recording, all partitioning is performed at record level: records are assigned to splits first, and windows inherit that assignment. For the Random Forest benchmark, grouped holdout evaluation follows a 70/30 train-test policy by record identifier, with class distribution monitored to retain clinically meaningful minority representation. Hyperparameter tuning and threshold selection are conducted using validation data derived only from the training side, ensuring the test split remains untouched until final reporting.

For the 1D-CNN benchmark, a grouped stratified three-way split is used with fixed ratios of 70% train, 15% validation, and 15% test at record level, controlled by a fixed random seed for reproducibility. Stratification is based on record outcome labels to reduce split drift between Normal, Suspicious, and Pathological groups. This provides a dedicated validation stream for early stopping and model selection while preserving a fully independent held-out test set for final comparison.

Class imbalance is handled through cost-sensitive learning rather than synthetic oversampling in the core pipeline. Random Forest uses built-in class weighting, and CNN optimization applies weighted cross-entropy derived from training-label frequencies. This avoids introducing synthetic temporal patterns while still penalizing minority-class errors during fitting. For this dissertation submission, final performance claims are based on fixed grouped holdout evaluation with split-level composition diagnostics and class-wise error analysis, rather than repeated grouped resampling. All split manifests, random seeds, class distributions, and tuning outputs are versioned as artefacts to enable exact reruns and independent audit of model selection decisions #cite(<francis2024_scoping>); #cite(<nazli2025_imbalanced>); #cite(<mendis2025_crossdb>).

Sensitivity analyses using alternative grouped seeds were not executed in this dissertation because the available compute/time budget was allocated to one fixed, leakage-aware benchmark cycle with full artefact traceability. This boundary is stated explicitly to keep validation claims proportional to executed experiments while preserving a clear path for stronger uncertainty quantification in later iterations. This framing matches the final results chapter.

== Evaluation Metrics

Evaluation is designed to reflect both statistical performance and clinical utility in imbalanced three-class CTG classification. The core reported metrics are overall accuracy, macro-F1, weighted-F1, balanced accuracy, class-wise precision and recall, and confusion matrices on held-out data. Accuracy provides a global summary, but it is not interpreted alone because majority-class prevalence can mask poor minority-class detection #cite(<francis2024_scoping>). Macro-F1 is prioritized for cross-model comparison because it weights classes equally and penalizes collapse on clinically important minority states. Weighted-F1 is also reported to reflect population-level behaviour under observed class proportions.

Class-wise recall, especially Pathological recall, is treated as a key safety-oriented indicator, while precision is used to assess false-alert burden relevant to intervention decisions. Confusion matrices are analyzed to identify systematic misclassification pathways, particularly Normal-Suspicious and Suspicious-Pathological overlap. Balanced accuracy complements these analyses by averaging class-wise true positive rates and reducing dominance from prevalent classes #cite(<nazli2025_imbalanced>). For probabilistic outputs, one-vs-rest ROC-AUC is reported where calibration and score distributions permit stable estimation.

Beyond point estimates, evaluation includes efficiency metrics (training time, inference time, model size) to support deployment realism. Metric definitions and formulas are fixed pre-analysis. This dissertation reports fixed-split results with class-wise diagnostics and does not claim inferential significance from repeated paired testing, because repeated grouped resampling was not executed in the final benchmark cycle. Future repeated runs can add non-parametric significance testing for partition-noise sensitivity #cite(<khare2022_compare>); #cite(<mendis2025_crossdb>).

== Interpretability Analysis

Interpretability analysis is used to explain why the benchmarked models produce specific CTG classifications and to assess whether those explanations are clinically plausible. For the Random Forest baseline, global feature importance is extracted directly from the fitted ensemble and summarized at both individual-feature and clinical-domain levels. This enables comparison of contribution from deceleration, contraction, variability, baseline, frequency, and complexity descriptors, rather than relying only on aggregate accuracy #cite(<feng2023_shap>); #cite(<francis2024_scoping>). Interpretation is coupled with confusion-matrix error analysis to identify where influential features align with frequent misclassification pathways, especially around Suspicious-Pathological boundaries.

For the 1D-CNN, interpretability is assessed through comparative behaviour rather than full attribution maps in this phase, with emphasis on class-wise recall patterns and failure cases relative to the feature-based baseline. This keeps conclusions grounded in implemented artefacts while avoiding over-claiming explanation quality. The analysis therefore prioritizes reproducible, auditable transparency: ranked feature lists, domain-importance summaries, and error cases. Detailed importance tables, domain mappings, and plots are provided in Appendix E for review.

== Implementation Details

Implementation uses Python with a modular `src/` layout (preprocessing, segmentation, feature extraction, CNN, utilities) and notebook blocks for experiments and reporting. Core dependencies are NumPy, pandas, SciPy, scikit-learn, WFDB, matplotlib, and PyTorch. Main runs used Linux, Python 3.10.19, CPU execution, fixed random seeds, and shared hardware/software conditions across RF and CNN. Efficiency metrics were measured as wall-clock training duration, per-window inference time, and saved model size. Reproducibility controls include record-level split manifests, JSON/CSV reports, and versioned output artefacts for metrics and figures. Runs are scripted end-to-end and timing logs are retained with artefact exports to support efficiency comparisons across model families. Shared preprocessing definitions were reused across RF and CNN to avoid implementation drift. Full environment and package versions are listed in Appendix A #cite(<chudacek2014_open>); #cite(<francis2024_scoping>).

= Results

== Data Characteristics

The results are based on 552 CTU-UHB recordings, with 550 retained for window-level modelling after excluding two records with unknown outcome labels. Recording duration averages 74.16 minutes (median 71.71, range 60.00-90.08), yielding over 682 monitored hours for analysis. Segmentation produced 35,787 candidate windows, of which 29,989 passed quality criteria and entered model development. At record level, the original cohort is predominantly Normal (447) with a smaller Pathological subset (103). After feature-based relabeling for the 3-class benchmark, the modelling distribution is skewed toward Pathological and Suspicious windows, reflecting conservative risk-labeling rules. This imbalance motivates the class-sensitive training and evaluation strategy. Overall, the dataset provides temporal coverage while retaining realistic variability in quality and outcome prevalence #cite(<chudacek2014_open>).

== Signal Quality Analysis

Signal quality assessment confirms that preprocessing removed artefactual contamination while preserving sufficient data for modelling. Using the predefined criterion of valid FHR percentage >= 50%, 542 of 552 records passed initial quality screening and 10 were excluded for severe signal limitations. Across the cohort, mean FHR missingness is 18.79% and mean FHR outlier rate is 1.53%, while uterine contraction channels show negligible missingness and low outlier burden (mean 0.69%). Median post-cleaning FHR validity is 80.02% with an interquartile range from 72.13% to 89.37%, supporting stable downstream window generation. Window-level filters then removed segments with long post-cleaning gaps, reducing interpolation-driven bias in later modelling. Residual variation contributes to borderline-class uncertainty, consistent with CTG data quality observations #cite(<francis2024_scoping>).

== Feature Extraction Results

Feature extraction generated a complete leakage-aware matrix with 29,989 windows and 49 columns, including metadata and 35 model features used for Random Forest training. The extracted set spans baseline, variability, acceleration/deceleration morphology, spectral bands, entropy measures, uterine contraction dynamics, and post-processing quality indicators. No extraction failures were recorded in the final run, and missingness across exported feature columns is effectively zero after controlled interpolation and finite-value handling. Event-derived descriptors captured clinically relevant burden statistics (for example, deceleration duration and area), while contraction features captured peak frequency, inter-peak interval, and aggregate UC activity. Frequency and complexity descriptors were also computed successfully across windows, enabling direct comparison between handcrafted physiologic representation and raw-signal learning. Feature domains remained numerically stable across splits, supporting downstream threshold tuning and error analysis without additional imputation or feature-pruning passes in the final benchmark run. This stage therefore confirms that the extraction pipeline is operationally robust at scale and suitable for reproducible benchmarking and interpretation workflows #cite(<hoodbhoy2019_ml>).

== Classifier Performance Comparison

#figure(
  table(
    columns: 4,
    [*Model*], [*Accuracy*], [*Macro-F1*], [*Pathological Recall*],
    [Random Forest (window)], [95.55%], [0.945], [96.76%],
    [CNN (window)], [65.31%], [0.584], [77.06%],
    [CNN (record)], [48.19%], [0.383], [81.58%],
  ),
  caption: [Classifier comparison on held-out data.],
)<tb:metrics>

In @tb:metrics, the primary benchmark outcomes show a clear gap between the feature-based Random Forest and the tested 1D-CNN under grouped evaluation. On held-out test windows, Random Forest achieves 95.55% accuracy and macro-F1 0.945, whereas CNN reaches 65.31% accuracy and macro-F1 0.584. The absolute deltas are -30.24 percentage points for accuracy and -0.361 for macro-F1 in favour of Random Forest. Weighted-F1 and balanced-accuracy trends follow the same direction in the detailed reports, indicating that the gap is not an artifact of one metric choice. Pathological recall remains comparatively high for both models (RF 96.76%, CNN 77.06%), but CNN exhibits broad confusion in Normal and Suspicious classes, reducing balanced clinical utility.

Per-class analysis reinforces this pattern. For CNN window-level predictions, Normal precision/recall are 0.444/0.570 and Suspicious precision/recall are 0.494/0.449, while Pathological precision/recall are 0.796/0.771. This indicates a classifier that preserves some high-risk sensitivity but struggles to maintain class boundaries for lower-risk states. In contrast, RF class-level metrics are consistently high across all classes, with fewer off-diagonal errors and tighter confusion structure. The RF confusion matrix shows small leakage from Normal into Suspicious/Pathological and controlled overlap between Suspicious and Pathological, consistent with the higher macro-F1 value.

Record-level aggregation further separates the models. CNN accuracy drops to 48.19% and macro-F1 to 0.383 when predictions are evaluated at record granularity, confirming instability after temporal aggregation. Record-level recalls of 0.143 (Normal), 0.250 (Suspicious), and 0.816 (Pathological) show strong bias toward high-risk assignment. Operationally, this would increase escalation tendency and likely produce substantial alert burden without additional calibration or threshold redesign. RF results, in contrast, remain consistent with the high-performing holdout profile reported in Block E and downstream Block F artefacts.

Split composition and training dynamics provide further context. RF uses 21,153 training windows and 8,836 grouped test windows (385 and 165 records respectively), while CNN uses 21,125/4,448/4,416 train/validation/test windows with 384/83/83 records. CNN early stopping selected best epoch 8 with best validation macro-F1 around 0.564, indicating that even optimized validation checkpoints did not close the gap to RF. Window-level CNN loss remained substantially higher than RF surrogate error rates implied by confusion structure, and this divergence persisted under identical preprocessing and record grouping constraints.

Efficiency evidence is made explicit in @tb:eff_explicit, with raw values archived in Appendix D. The deployed RF artifact is substantially larger (18,396,698 bytes) than the CNN checkpoint (155,811 bytes), while model structure shows the opposite pattern in operational complexity: RF stores 250 trees with 207,860 total nodes (mean depth 12), whereas the CNN contains 36,774 trainable parameters. In other words, RF delivers stronger predictive performance at the cost of larger on-disk footprint, while CNN is compact on disk but underperforms in current discrimination metrics. These trade-offs are directly relevant for deployment planning where storage, update distribution, and runtime behavior must be balanced.

#figure(
  table(
    columns: 4,
    [*Model*], [*Artifact Size*], [*Complexity Evidence*], [*Training Evidence*],
    [Random Forest], [18,396,698 bytes (~17.55 MiB)], [250 trees; 207,860 total nodes; mean depth 12], [single-fit estimator (no epoch schedule)],
    [1D-CNN], [155,811 bytes (~152.2 KiB)], [36,774 trainable parameters], [best epoch 8; stopped by epoch 18 of planned 40],
  ),
  caption: [Explicit efficiency evidence from saved model artifacts and training reports.],
)<tb:eff_explicit>

A secondary comparison using the `cnn_vs_rf_comparison.json` artefact confirms the same metric ordering outside the main reporting notebook, supporting internal consistency across generated outputs. Threshold analysis also favours RF in this benchmark: pathological operating points can be shifted for sensitivity or balanced behaviour without collapsing minority precision, whereas CNN threshold moves show less stable class separation between Suspicious and Pathological cases.

These findings are presented as fixed-split comparative evidence, not as inferential ranking proof from repeated-resampling statistics. Both models share leakage-aware record grouping and common preprocessing definitions, so the observed gap is unlikely to be explained by split contamination or pipeline mismatch. Under this dataset size and weak-label setting, engineered physiologic features appear better aligned to separability than raw-window deep representation learning. CNN remains a relevant development baseline, but it does not outperform the interpretable RF reference on clinically decisive metrics #cite(<mendis2025_crossdb>); #cite(<chiou2025_npj>); #cite(<francis2024_scoping>).

== Statistical Analysis

Statistical interpretation in this chapter emphasizes effect magnitude, class-wise consistency, and agreement across evaluation views rather than formal inferential testing. On held-out windows, the RF-CNN gap is large (accuracy +30.24 points; macro-F1 +0.361 for RF), and improvements are not confined to the dominant class: RF improves Normal and Suspicious discrimination while retaining high Pathological recall. These descriptive gaps are operationally meaningful in this dataset because class-confusion patterns directly affect escalation burden and alert quality in downstream workflow policy.

Balanced-accuracy and confusion analyses support the same direction. CNN errors cluster in non-pathological boundaries and show bias toward Pathological prediction, while RF errors are fewer and more localized. Record-level aggregation provides an additional stress test: CNN drops from 65.31% to 48.19% accuracy and from 0.584 to 0.383 macro-F1, indicating weaker robustness when predictions are evaluated at clinically meaningful granularity.

Generalisability claims are bounded to within-cohort evidence. All partitions are grouped by record identifier to prevent leakage, split compositions are reported, and no external cohort is used in this chapter. Because results are fixed-split benchmark artefacts, formal paired significance testing is not claimed here; repeated-resampling inference is outside the executed evaluation scope. Conclusions therefore rely on multi-metric convergence, reproducible split diagnostics, and consistent effect direction across window-level and record-level outputs #cite(<khare2022_compare>); #cite(<m2025_icitiit>).

== Interpretability Analysis

Interpretability outputs from the Random Forest model indicate that physiologically plausible domains dominate prediction decisions. Feature-level rankings are led by contraction and deceleration descriptors: `uc_peak_count` is the top feature, followed by `decel_total_dur_sec`, `decel_mean_dur_sec`, `uc_mean_interpeak_sec`, and `decel_area_bpm_sec`. Domain-level aggregation confirms this emphasis, with Decelerations contributing 33.67% of cumulative importance and Contractions 32.64%. Variability (10.36%), Accelerations (8.46%), and Baseline FHR (6.46%) contribute meaningful secondary signal, while Frequency-domain and Complexity features provide smaller but non-zero support.

This distribution is clinically coherent because contraction context and deceleration burden are central to intrapartum risk interpretation. The model therefore appears to rely on interpretable physiology-linked structure rather than opaque single-feature heuristics. Error-linked review also shows that ambiguity is concentrated around Suspicious boundaries rather than obvious Pathological cases, suggesting that residual uncertainty reflects intrinsic class overlap more than feature irrelevance.

For the CNN pathway, interpretability in this phase is limited to comparative failure profiling and class-wise behaviour, not full attribution maps. This conservative reporting avoids over-claiming explanation quality for a model that currently underperforms the RF baseline. Overall, the interpretability analysis strengthens confidence in RF as the safer clinician-facing option for this dataset and protocol. Results are consistent with emphasis on explainable CTG modelling in recent applied studies #cite(<feng2023_shap>); #cite(<ribeiro2016_lime>).

== Visualisation

#figure(
  image("../outputs/models/rf_3class_confusion_matrix.png", width: 80%),
  caption: [Random Forest 3-class confusion matrix on held-out test windows.],
)<fig:rf_cm>

#figure(
  image("../outputs/models/cnn_3class_confusion_matrix.png", width: 80%),
  caption: [CNN 3-class confusion matrix on held-out test windows.],
)<fig:cnn_cm>

#figure(
  image("../outputs/models/rf_3class_feature_importance.png", width: 80%),
  caption: [Random Forest feature importance ranked by contribution.],
)<fig:rf_fi>

#figure(
  image("../outputs/models/blockF_alert_timeline.png", width: 90%),
  caption: [Temporal alert timeline after causal smoothing policy.],
)<fig:alerts>

Visual outputs are used to connect quantitative metrics to clinically interpretable behaviour. In @fig:rf_cm, the Random Forest confusion matrix shows strong diagonal dominance with limited off-diagonal spillover. In @fig:cnn_cm, off-diagonal mass is substantially larger, especially from Normal and Suspicious into Pathological predictions. This visual contrast mirrors the macro-F1 gap in @tb:metrics and clarifies why CNN record-level performance degrades after aggregation.

Additional plots contextualize data and model dynamics. The class-distribution panel illustrates imbalance in both source outcomes and derived 3-class labels, helping explain why macro-level metrics are prioritized over accuracy alone. In @fig:rf_fi, domain concentration around contraction and deceleration variables is consistent with the interpretability summary. Threshold plots show how changing pathological probability cutoffs modifies sensitivity-specificity trade-offs, linking model scores to operational policy.

For deployment-oriented analysis, @fig:alerts provides temporal context beyond static confusion matrices. Causal smoothing reduces transition frequency from 8.22 to 4.74 transitions per hour, improving alert stability, but also alters false-alert behaviour and lowers raw classification agreement in the smoothed stream. These results highlight a key implementation trade-off: temporal consistency can improve bedside usability while shifting metric balance.

Together, the visualization suite provides complementary evidence for model selection by showing where errors occur, how they evolve over time, and how thresholding/smoothing policies affect practical decision support. Extended plots, per-class distribution panels, CNN training curves, and alert diagnostics are provided in Appendix E for reproducibility and independent review #cite(<francis2024_scoping>); #cite(<mendis2025_crossdb>). Figures in this chapter are generated directly from versioned output artefacts to ensure exact reproducibility between narrative interpretation, tables, and plotted evidence across review processes. Separate training-curve visualization for CNN shows rapid early validation plateau and later overfitting tendency, supporting early-stopping selection and explaining limited generalization gains. Combined with RF threshold and feature-importance figures, these visuals provide a compact but complete evidence chain from data distribution to deployment policy for reviewers.

= Discussion

== Interpretation of Results

The benchmark results indicate that model performance in this CTG setting is driven less by model complexity and more by alignment between data regime and representation strategy. Random Forest outperformed the tested 1D-CNN across window-level and record-level views, with higher accuracy, macro-F1, and tighter confusion structure. A plausible reason is that RF is trained on denoised, clinically structured summaries (baseline, variability, deceleration burden, contraction morphology) that encode signal characteristics used in obstetric interpretation. In contrast, CNN must infer these abstractions from raw windows under weaker supervision and limited cohort diversity, which increases sensitivity to nuisance variation and class-prior bias. This mismatch is visible in the CNN error profile: moderate Pathological sensitivity is retained, but specificity and boundary discrimination degrade in Normal and Suspicious classes, increasing over-escalation risk.

A key interpretation is that weakly supervised labels and label distribution asymmetry penalize deep representation learning more than feature-based methods in this dataset. Window labels are inherited from record-level outcomes, so many local segments carry ambiguous supervision; RF is relatively robust because engineered features compress windows into stable statistics, reducing short-term noise. CNN, by contrast, optimizes directly on high-dimensional raw sequences and therefore pays a larger penalty when local morphology and inherited labels are weakly aligned. The training dynamics are consistent with this mechanism: validation macro-F1 plateaus early, then stops improving despite additional epochs, indicating representation saturation rather than simple undertraining. Grouped record-level splitting supports the interpretation that these effects are model-data interactions rather than leakage artefacts.

The threshold and post-processing findings add further nuance. Temporal smoothing improves alert stability but introduces trade-offs in raw classification agreement, reinforcing that bedside utility depends on calibration policy, not only classifier architecture. Overall, the results support a staged strategy: use interpretable tabular models as the operational baseline, then revisit deep models when larger cohorts, stronger event-level labels, and cross-database validation are available #cite(<francis2024_scoping>); #cite(<mendis2025_crossdb>); #cite(<chiou2025_npj>).

Interpretation is intentionally bounded by the executed evaluation design. These conclusions are drawn from fixed-split, leakage-aware grouped benchmarking with internally consistent artefacts, not from repeated-resampling inferential testing. Accordingly, model ranking statements in this chapter should be read as strong comparative evidence within this cohort and protocol, rather than universal claims across hospitals or data regimes. Even with that boundary, the effect sizes are large and directionally stable across window-level metrics, record-level aggregation, confusion structure, and threshold-behaviour analysis, which makes the practical recommendation defensible for current decision-support prototyping.
For communication consistency, this bounded interpretation is maintained across Results, Discussion, and Conclusion so narrative confidence stays aligned with the statistical scope of executed analyses.

== Benchmarking Insights

Several benchmarking insights emerge from this study. First, leakage control is not optional in CTG window classification. When many windows originate from the same labour recording, any split strategy that ignores record boundaries can inflate performance by allowing near-duplicate temporal context into both training and test sets. The grouped split policy used here therefore functions as a primary validity safeguard.

Second, metric choice materially affects interpretation. High overall accuracy alone would obscure clinically relevant failure modes, particularly in Suspicious and boundary-pathological cases. Combining macro-F1, class-wise recall/precision, balanced accuracy, and confusion analysis provides a more faithful picture of model behavior under imbalance. In this benchmark, these metrics converge on the same ordering.

Third, representation quality currently outweighs model depth. The engineered feature set preserves clinically discriminative structure (event burden, duration/amplitude descriptors, contraction timing proxies) while discarding nuisance variance in raw traces. That gives RF a favourable bias-variance trade-off under finite data and weak labels. CNN has higher representational capacity, but in this regime that capacity is underconstrained: it must learn denoising, feature abstraction, and decision boundaries simultaneously from raw windows, which increases instability in boundary classes. This does not imply deep models are unsuitable in principle; it indicates mismatch between current dataset characteristics and current learning conditions.

Fourth, operational benchmarking must include efficiency and temporal behavior. Training cost, inference practicality, threshold sensitivity, and smoothing effects all influence deployability. A model with lower macro-F1 but faster runtime may still be attractive in some settings, while a high-scoring model that produces unstable alerts may not be clinically acceptable. These findings reinforce benchmarking as a multi-objective exercise spanning discrimination, robustness, and usability #cite(<khare2022_compare>); #cite(<nazli2025_imbalanced>); #cite(<francis2024_scoping>).

Finally, benchmarking should preserve traceability from preprocessing through post-processing, and should treat threshold/smoothing policy as part of evaluation rather than as a separate deployment afterthought. In this project, stage-wise artefacts allow disagreements to be traced to concrete inputs, features, probabilities, and alert rules, reducing interpretive ambiguity during governance review.
Together, these insights support practical model-selection decisions while keeping comparative claims explicitly tied to this fixed-split benchmark design.

== Clinical Implications

The clinical implication of these findings is that an interpretable feature-based decision-support pathway is currently the more suitable candidate within this benchmark setting for intrapartum assistance. RF predictions are better calibrated to class boundaries and produce fewer problematic cross-class confusions than the tested CNN, especially between Normal and Suspicious windows where unnecessary escalation risk is high. This supports use of RF-derived probabilities as advisory signals for clinician review rather than autonomous decisions.

Results from temporal post-processing further suggest that alert design must be treated as a clinical intervention parameter. Causal smoothing can reduce transition noise and improve usability during continuous monitoring, but threshold choices can shift false-alert burden and escalation rates. Therefore, deployment should pair classifier outputs with explicit governance rules: threshold rationale, escalation criteria, and audit logging for retrospective review.

Importantly, this study does not replace clinical judgement. It provides quantitative evidence for where ML can support consistency and where uncertainty remains concentrated, particularly around Suspicious cases. A realistic clinical pathway is phased adoption: shadow-mode validation, multidisciplinary threshold review, then monitored integration with feedback loops. This approach aligns technical performance with patient safety and workflow acceptability in labour ward settings #cite(<figo2015_intro>); #cite(<feng2023_shap>).

Clinically, this means implementation should prioritize decision-confidence communication, not only class labels. Probability outputs, top contributing feature domains, and recent temporal trajectory should be displayed together so clinicians can contextualize recommendations quickly during labour.

The findings also support targeted use cases. The model may add most value during prolonged monitoring periods, handover transitions, and ambiguous trace evolution where consistency is difficult to maintain manually over time during high workload periods.

== Comparison with State-of-the-Art

Compared with recent CTG literature, this study’s results are directionally consistent with reports that tabular baselines can remain competitive when datasets are moderate in size, labels are weakly supervised, and interpretability is a deployment requirement. Published deep-learning studies frequently report higher headline performance, but many also rely on different split policies, larger cohorts, or label definitions that are not directly comparable. Under the leakage-aware grouped protocol used here, the tested 1D-CNN does not match RF performance, particularly on macro-level balance and record-level stability.

This contrast should be interpreted as contextual, not universal. Deep models can outperform traditional methods in settings with richer annotations, larger multi-site training pools, and robust external validation. However, cross-database work also shows substantial performance degradation when distribution shift is introduced, which supports caution in transferring single-cohort gains to practice.

The present findings therefore contribute to state-of-the-art discourse by emphasizing controlled benchmarking discipline over architecture novelty. Rather than claiming a globally superior model class, the dissertation identifies which model family is best-performing in this benchmark under specified constraints: CTU-UHB data, weak labels, grouped splitting, and clinically interpretable decision support requirements. This evidence complements, rather than contradicts, broader deep-learning progress by clarifying boundary conditions where interpretable models remain the stronger operational choice #cite(<francis2024_scoping>); #cite(<mendis2025_crossdb>); #cite(<chiou2025_npj>).

It also extends prior work by integrating efficiency and alert-behaviour considerations into comparative interpretation. Many published comparisons stop at discrimination metrics, whereas this study evaluates whether models remain usable under practical thresholding and temporal-smoothing policies. That emphasis better reflects deployment reality and helps explain why nominally promising architectures may underperform in clinical workflows.

Accordingly, the contribution to state-of-the-art is methodological as well as empirical: it presents a reproducible evaluation template reusable across future cohorts and model updates, enabling clearer longitudinal comparisons than isolated single-metric reports. This framing supports fairer interpretation across studies using different data scales, label granularity, and operational constraints in maternity settings.

It also encourages explicit reporting of split policy, threshold policy, and post-processing policy as benchmark variables, which improves reproducibility and cross-study comparability for clinically oriented decision-support research. A compact cross-study matrix is provided in Appendix G.

== Trustworthy AI Considerations

Trustworthy AI considerations in this project center on transparency, controllability, and governance readiness rather than accuracy alone. The selected RF baseline supports direct feature-importance inspection and domain-level attribution, enabling clinicians and auditors to understand which physiological patterns most influenced predictions. This interpretability is essential in high-stakes intrapartum use where recommendations must be explainable and contestable.

A second consideration is calibration and policy transparency. Thresholds for Pathological and Suspicious escalation materially change downstream alert behavior; therefore, threshold settings must be documented, justified, and periodically reviewed against outcome data. The project’s versioned artefacts and split manifests support this by enabling reproducible audits of model behavior across revisions.

A third consideration is human oversight. The system is framed as decision support, not decision replacement, and should be integrated with explicit clinician-in-the-loop protocols. This includes escalation accountability, fallback actions when confidence is low, and governance pathways for drift detection.

Together, these elements align with practical trustworthy-AI principles: interpretable outputs, reproducible processes, and bounded automation. They also reduce medico-legal and operational risk during early deployment phases while preserving a pathway for iterative model improvement #cite(<feng2023_shap>); #cite(<ribeiro2016_lime>); #cite(<francis2024_scoping>).

Data governance is an additional trust pillar. Even when model logic is interpretable, deployment requires strict handling of version control, audit trails, and access boundaries for clinical data and prediction artefacts. Embedding these controls early supports responsible scaling and regulatory alignment.

Trust also depends on communication design: explanations must be concise, framed, and available at the point of decision.

== Limitations

This study has several limitations. First, labels are weakly supervised at window level because outcome information is primarily record-level; this can introduce label noise when local signal segments do not fully reflect final delivery outcomes. Second, the work is based on a single primary dataset for model development, which constrains external validity despite internal leakage controls. Third, class structure is imbalanced and clinically asymmetric, which can bias optimization and threshold behavior even with class weighting.

Methodologically, only one deep architecture family was benchmarked against the RF baseline. Although this matches project scope and supports controlled comparison, it does not exhaust possible deep-learning variants, augmentation strategies, or calibration schemes. Computational constraints also limited broader hyperparameter exploration and repeated large-scale resampling experiments.

From a clinical perspective, the evaluation is retrospective and does not include prospective workflow testing with obstetric teams. Therefore, usability, alarm fatigue effects, and human factors integration are inferred from proxies rather than measured directly.

Finally, the current implementation focuses on CTG-only signals and does not integrate multimodal covariates such as maternal clinical context or fetal ECG. As a result, conclusions should be interpreted as within-scope comparative evidence, not as a final statement on all intrapartum AI decision-support designs #cite(<chudacek2014_open>); #cite(<mendis2025_crossdb>).

Another limitation is potential labeling drift introduced by proxy rules and derived class mapping. If rule thresholds do not reflect bedside interpretation across contexts, learned patterns may overfit labeling conventions rather than physiology. Uncertainty in model ranking is not fully characterized in this fixed-split report and is not supported by repeated-resampling inference.

== Threats to Validity

Key threats to validity fall into internal, construct, and external categories. Internally, residual preprocessing artefacts and weak window labels may confound learned associations, especially near class boundaries. Although grouped splitting reduces leakage risk, it cannot fully remove dependence introduced by shared recording context and outcome-linked labeling assumptions.

Construct validity is threatened by the gap between benchmark labels and true physiological state at every time point. Metrics such as macro-F1 and balanced accuracy are appropriate for imbalanced classification, but they remain proxies for clinical utility rather than direct measures of improved neonatal outcomes. Similarly, record-level aggregation choices and smoothing policies can alter interpretation of model benefit.

External validity is limited by cohort scope, acquisition practices, and potential distribution shift across hospitals, devices, and clinical protocols. Performance observed on CTU-UHB may not transfer unchanged to other populations without recalibration and revalidation. This is particularly relevant for deep models, where representation sensitivity to data domain can be high.

To mitigate these threats, the study emphasizes reproducible artefacts, transparent split policies, multi-metric reporting, and cautious claims. Future work should include cross-site validation, prospective evaluation, and clinician-centered usability assessment to strengthen causal and practical confidence in deployment recommendations #cite(<francis2024_scoping>); #cite(<mendis2025_crossdb>).

There is also a reporting-validity threat if stakeholders overgeneralize from aggregate metrics without reviewing class-specific behavior and alert dynamics. The dissertation mitigates this with confusion structures and temporal summaries, but interpretation discipline remains essential. Temporal non-stationarity in clinical practice may also reduce future validity without ongoing recalibration and monitoring, especially across devices and care pathways over time.

= Conclusion and Future Work

== Summary of Findings

This dissertation benchmarked feature-based and deep-learning CTG classifiers under leakage-aware, record-grouped evaluation. Random Forest outperformed the tested 1D-CNN at both window and record levels, with higher accuracy, macro-F1, and more coherent confusion structure (@tb:metrics; @fig:rf_cm; @fig:cnn_cm). The observed gap indicates that engineered physiologic descriptors currently capture discriminative signal more effectively than raw-window representation learning in this cohort.

Interpretation also depends on pipeline policy, not only architecture. Threshold selection, class imbalance handling, and temporal smoothing materially changed operational behaviour, particularly alert stability and escalation burden (@fig:alerts). Benchmark conclusions therefore combine discrimination metrics with deployment-facing indicators, including confusion pathways and post-processing effects.

Interpretability outputs were clinically plausible: contraction and deceleration domains contributed most strongly to Random Forest decisions, while residual uncertainty concentrated around Suspicious boundaries (@fig:rf_fi). This pattern aligns with intrapartum reasoning and supports RF as an appropriate translational baseline in this cohort for clinician-facing decision support.

Overall, the findings support a staged strategy: deploy interpretable tabular models first, then revisit deep models when stronger event-level labels, larger cohorts, and cross-database validation are available #cite(<francis2024_scoping>); #cite(<mendis2025_crossdb>); #cite(<chiou2025_npj>).

Reproducibility infrastructure supports these conclusions: fixed seeds, preserved split manifests, and exported reports enabled direct auditing across notebooks and artefacts (Appendix A to Appendix F).

== Contributions

This work contributes an end-to-end CTG benchmarking pipeline that is reproducible, leakage-aware, and clinically interpretable. Methodologically, it provides fair comparison of model families under shared preprocessing, grouped splitting, and unified metric reporting. Empirically, it reports that the implemented RF baseline outperforms the tested 1D-CNN in this data regime (@tb:metrics; @fig:rf_cm; @fig:cnn_cm). Practically, it links model evaluation to deployment-facing outputs, including threshold policies, temporal alert behaviour, and auditable artefacts (@fig:alerts; @tb:eff_explicit). It also provides reusable reporting templates and artefact structures that lower replication barriers for future cohorts and follow-on validation work #cite(<feng2023_shap>).
Because claims are bounded to within-cohort fixed-split evidence, these contributions should be interpreted as a rigorous current operational baseline and reusable evaluation scaffold, rather than a definitive cross-site ranking of model families.

== Practical Recommendations

Use the Random Forest pipeline as the reference model for clinical decision support experiments, with thresholds tuned to stated safety objectives and reviewed by stakeholders (@tb:metrics; @fig:rf_cm).

Maintain grouped record-level split policies, class-aware metrics, and versioned artefact logging as mandatory benchmarking controls for all future model updates.

Use temporal smoothing as a configurable post-processing layer rather than a fixed default, and report its impact on transitions, false alerts, and escalation behaviour before deployment decisions (@fig:alerts).

Prioritize explainable output presentation in user interfaces, including probability context and top contributing feature domains, so recommendations remain interpretable at the point of care and support transparent clinician override (@fig:rf_fi).

== Future Research Directions

Future research should prioritize external validity, richer supervision, and prospective workflow evaluation. First, the benchmark should be extended to multi-site cohorts with harmonized acquisition metadata to quantify distribution-shift effects and improve transferability across hospitals and devices. Second, label quality should be strengthened through event-level annotation and clinician-reviewed segments, enabling finer-grained learning targets than record-outcome proxies. Third, deep architectures should be revisited under these improved conditions, including calibrated CNN variants and hybrid feature-plus-signal models, with evaluation framed by the same leakage-aware protocol to preserve comparability.

A parallel priority is multimodal integration. Incorporating maternal covariates, intervention context, and complementary fetal signals may improve ambiguity handling in Suspicious cases and reduce over-escalation. Finally, prospective clinical studies are required to assess bedside usability, alert fatigue, response-time effects, and governance fit under real operating constraints. This should include human factors analysis, threshold review workflows, and drift monitoring plans so that technical gains translate into safe and sustainable clinical adoption #cite(<chiou2025_npj>); #cite(<mendis2025_crossdb>).

Future studies should include repeated grouped resampling with confidence intervals and significance testing to better quantify ranking uncertainty across models. Adaptive threshold governance should also be evaluated, with periodic recalibration against monitored outcomes as data distributions evolve over time. Prospective implementation studies should assess workflow fit, alert communication design, and governance burden during use. Review boards should be involved from protocol design to deployment audit; a practical implementation checklist is included in Appendix H.

== Closing Statement

Cardiotocography remains clinically indispensable yet methodologically demanding, and this dissertation addresses that tension through rigorous, reproducible benchmarking. The central message is straightforward: trustworthy progress in CTG AI depends on disciplined evaluation design as much as model innovation. By combining preprocessing, leakage-aware comparison, multi-metric reporting, and interpretable analysis, the project provides evidence that is credible and clinically actionable (@tb:metrics; @tb:eff_explicit; @fig:rf_fi).

The resulting baseline is not presented as final truth, but as a foundation for iterative improvement. That foundation now supports the next phase: broader validation, stronger labels, multimodal enrichment, and prospective clinical integration. In that sense, the contribution is dual-purpose: immediate guidance for decision-support development and a framework for future research. This is the pathway from benchmark performance to impact in intrapartum care.

#set heading(numbering: none)

#heading(level: 1, numbering: none)[Appendices]

#heading(level: 2, numbering: none)[Appendix A] <app:a>

== Runtime Environment

#table(
  columns: 2,
  [*Item*], [*Value*],
  [Python], [3.10.19],
  [Platform], [Linux-6.6.87.2-microsoft-standard-WSL2-x86_64],
  [CPU], [Intel(R) Core(TM) Ultra 7 265T (20 cores)],
  [RAM], [15 GiB],
  [NumPy], [2.2.6],
  [pandas], [2.3.3],
  [SciPy], [1.15.2],
  [scikit-learn], [1.7.2],
  [joblib], [1.5.3],
  [matplotlib], [3.10.8],
  [seaborn], [0.13.2],
  [wfdb], [4.3.1],
  [PyTorch], [2.10.0+cu128],
)<tb:app_env>

== Project Structure

- `src/preprocessing.py`: signal cleaning and QC
- `src/segmentation.py`: window creation and keep-window policy
- `src/feature_extraction.py`: 35-feature extraction pipeline
- `src/cnn.py`: grouped split, CNN training, and reporting
- `outputs/models/`: reports, plots, and model artefacts

== Repository Access

- GitHub repository: `https://github.com/Naem-Haq/CTG`
- Clone URL: `https://github.com/Naem-Haq/CTG.git`
- Submission baseline commit: `0f75020`

== Reproducible Run Sequence

```bash
python -m src.preprocessing
python -m src.segmentation
python -m src.feature_extraction
python -m src.cnn
```

#heading(level: 2, numbering: none)[Appendix B] <app:b>

== Preprocessing Configuration (Extract)

```python
@dataclass(frozen=True)
class PreprocessConfig:
    fs_default: float = 4.0
    fhr_min: float = 80.0
    fhr_max: float = 240.0
    fhr_missing_sentinel: float = 0.0
    uc_min: float = 0.0
    uc_max: float = 100.0
    max_interp_gap_sec: float = 30.0
    gauss_sigma_sec: float = 1.5
    zscore_per_record: bool = False
```

== Quality Summary (Dataset-Level)

- total recordings: 552
- passed quality criterion (`valid_fhr_pct >= 50%`): 542
- failed quality criterion: 10
- mean FHR missingness: 18.79%
- mean FHR outliers: 1.53%
- mean UC outliers: 0.69%

Failed record IDs: `1058`, `1164`, `1173`, `1198`, `1361`, `1431`, `2006`, `2013`, `2025`, `2030`.

#heading(level: 2, numbering: none)[Appendix C] <app:c>

== Feature Configuration (Extract)

```python
@dataclass(frozen=True)
class FeatureConfig:
    window_min: int = 10
    accel_thresh_bpm: float = 15.0
    accel_min_dur_sec: float = 15.0
    decel_thresh_bpm: float = 15.0
    decel_min_dur_sec: float = 15.0
    psd_nperseg_sec: int = 60
    perm_order: int = 3
    perm_delay: int = 1
    sampen_m: int = 2
    sampen_r: float = 0.2
    sampen_downsample_to_hz: float = 1.0
    uc_peak_distance_sec: float = 75.0
    uc_prom_scale: float = 0.3
```

== Feature Groups Used in Modelling

- baseline and central tendency: `fhr_baseline_median`, `fhr_mean`, `fhr_median`
- variability: `fhr_std`, `fhr_mad`, `fhr_iqr`, `fhr_std_1min_means`
- acceleration/deceleration morphology: counts, durations, amplitudes, area
- frequency and complexity: bandpower, spectral entropy, permutation entropy, sample entropy
- uterine contraction dynamics: peak count, inter-peak interval, peak prominence, AUC
- quality indicators: `fhr_remaining_nan_pct`, `fhr_valid_pct_post`

Final matrix dimensions: 29,989 windows × 49 columns (metadata + 35 model features).

#heading(level: 2, numbering: none)[Appendix D] <app:d>

== Random Forest Final Configuration

#table(
  columns: 2,
  [*Parameter*], [*Value*],
  [`n_estimators`], [250],
  [`max_depth`], [12],
  [`min_samples_split`], [10],
  [`min_samples_leaf`], [5],
  [`class_weight`], [`balanced_subsample`],
)<tb:app_rf_cfg>

== CNN Final Configuration (Report)

#table(
  columns: 2,
  [*Parameter*], [*Value*],
  [`batch_size`], [64],
  [`epochs`], [40],
  [`learning_rate`], [0.0003],
  [`weight_decay`], [0.0001],
  [`dropout`], [0.3],
  [`train/val/test`], [0.70 / 0.15 / 0.15],
)<tb:app_cnn_cfg>

== Threshold Policy (RF)

- default: 0.50
- high-sensitivity pathological: 0.75
- balanced: 0.45
- high-specificity pathological: 0.30
- suspicious policy (Block F): 0.50

Primary artefacts: `outputs/models/rf_3class_report.json`, `outputs/models/cnn_3class_report.json`, `outputs/models/cnn_vs_rf_comparison.json`.

== Model Efficiency Evidence (Raw)

#table(
  columns: 3,
  [*Artifact*], [*Raw Value*], [*Interpretation*],
  [`rf_3class_model.joblib`], [18,396,698 bytes], [larger footprint, high predictive performance],
  [`cnn_3class_model.pt`], [155,811 bytes], [compact footprint, lower predictive performance in this benchmark],
  [RF structural complexity], [250 trees, 207,860 total nodes], [high ensemble capacity],
  [CNN structural complexity], [36,774 trainable parameters], [small neural model capacity],
)

#heading(level: 2, numbering: none)[Appendix E] <app:e>

== Extended Figures

#figure(
  image("../outputs/models/class_distribution.png", width: 80%),
  caption: [Class distribution used in modelling.],
)

#figure(
  image("../outputs/models/feature_distributions_by_class.png", width: 85%),
  caption: [Selected feature distributions by class.],
)

#figure(
  image("../outputs/models/cnn_3class_training_curves.png", width: 85%),
  caption: [CNN training and validation curves.],
)

#figure(
  image("../outputs/models/rf_3class_thresholds.png", width: 80%),
  caption: [RF threshold behavior for decision-policy tuning.],
)

Additional diagnostic outputs are stored under `outputs/models/` and include split summaries, confusion matrices, and alert timeline artefacts.

#heading(level: 2, numbering: none)[Appendix F] <app:f>

== Metric Definitions

```text
Accuracy = (TP + TN) / (TP + FP + TN + FN)
Precision = TP / (TP + FP)
Recall (Sensitivity) = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Balanced Accuracy = mean(class-wise recall)
```

== Reproducibility Checklist

- fixed random seed (`42`) across train/split routines
- record-grouped split manifests saved with outputs
- all core reports exported as JSON/CSV
- figure outputs generated from versioned artefacts
- same preprocessing definitions used for RF and CNN comparisons

== Statistical Reporting Notes

This dissertation reports fixed-split benchmark effects with class-wise diagnostics and deliberately avoids inferential significance claims from single-run comparisons, because repeated-resampling analysis is outside the executed evaluation scope.

#heading(level: 2, numbering: none)[Appendix G] <app:g>

== State-of-the-Art Comparison Matrix (Summary)

#table(
  columns: 4,
  [*Study*], [*Model Focus*], [*Validation Scope*], [*Interpretability*],
  [This dissertation], [RF vs 1D-CNN], [Grouped holdout + record-level view], [Feature/domain importance + error profiling],
  [Francis et al. 2024], [Scoping review], [Cross-study synthesis], [Highlights XAI need],
  [Mendis et al. 2025], [Deep models], [Cross-database], [Generalisability emphasis],
  [Chiou et al. 2025], [Deep CTG interpretation], [Development/evaluation study], [Model analysis reported],
)<tb:app_sota>

== Governance Checklist (Operational)

- threshold policy documented and versioned
- clinician override preserved
- model update audit trail maintained
- drift-monitoring plan defined
- alert burden metrics reviewed pre-deployment

#heading(level: 2, numbering: none)[Appendix H] <app:h>

== Prospective Validation Roadmap

1. Shadow-mode deployment on new cohort (no clinical actioning).
2. Weekly governance review of alerts, false alarms, and missed events.
3. Threshold recalibration against monitored outcomes.
4. Human factors assessment with obstetric and midwifery teams.
5. Controlled pilot with explicit escalation protocol.

== Stakeholder Review Checklist

- clinical safety lead sign-off
- data governance and privacy approval
- technical reproducibility re-run completed
- model-card/update note completed
- rollback/fail-safe pathway tested

#bibliography("refs.bib", title: [Bibliography], full: true)
