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
  abstract: [
    Cardiotocography (CTG) remains the primary clinical tool for monitoring fetal wellbeing during labour by recording fetal heart rate (FHR) and uterine contractions. However, interpretation of CTG traces is often subjective, leading to inconsistent clinical decisions and variable outcomes. This project explores the application of biomedical signal processing and supervised machine learning to support clinicians in the detection and classification of fetal heart rate patterns, with a focus on deceleration events indicative of potential fetal distress.

    The interim phase of this study establishes the theoretical foundation and methodological design for benchmarking machine learning classifiers in automated CTG interpretation. A comprehensive literature review highlights the limitations of manual CTG assessment, existing computational approaches, and the need for standardized benchmarking. The methodology outlines a reproducible pipeline involving signal preprocessing for artifact removal, feature extraction in time and frequency domains, and comparative evaluation of classifiers such as Support Vector Machines, Random Forests, and Gradient Boosting.

    This work aims to identify robust, interpretable models capable of enhancing clinical decision-making in intrapartum monitoring. Future stages will involve experimental benchmarking, statistical analysis of model performance, and interpretability evaluation using explainable AI techniques to ensure clinical transparency and trust.
  ],
  keywords: [Cardiotocography; Fetal monitoring; Machine learning; Random forest; Convolutional neural network; Clinical decision support],
  acknowledgments: [
    I would like to express my sincere gratitude to my supervisor, Dr. Salaheddin Alakkari, for his continuous guidance, insightful feedback, and support throughout the development of this project. I am also thankful to the INFANT Research Centre and the School of Engineering at University College Cork for providing the resources and expertise that have made this research possible.

    My appreciation extends to the clinicians and researchers involved in the AI4Life project, whose work and discussions have inspired many aspects of this study. Finally, I would like to thank my family and peers for their encouragement and support during this stage of my academic journey.
  ],
  acronyms: (
    "CTG": "Cardiotocography",
    "FHR": "Fetal Heart Rate",
    "UC": "Uterine Contractions",
    "MHR": "Maternal Heart Rate",
    "FIGO": "International Federation of Gynecology and Obstetrics",
    "RF": "Random Forest",
    "CNN": "Convolutional Neural Network",
    "SVM": "Support Vector Machine",
    "AUC": "Area Under the ROC Curve",
    "ROC": "Receiver Operating Characteristic",
    "SHAP": "SHapley Additive exPlanations",
    "LIME": "Local Interpretable Model-agnostic Explanations",
    "ML": "Machine Learning",
    "AI": "Artificial Intelligence",
  ),
)

= Introduction

#heading(level: 1, numbering: none)[Declaration]

I herewith declare that I have produced this paper without the prohibited assistance of third parties and without making use of aids other than those specified; notions taken over directly or indirectly from other sources have been identified as such. This paper has not previously been presented in identical or similar form to any other Irish or foreign examination board.

The thesis work was produced under the supervision of Dr. Salaheddin Alakkari at University of Limerick.

Limerick, 2026

#heading(level: 1, numbering: none)[AI Declaration]

I herewith declare that I have used artificial intelligence to produce my project and/or report in the following ways:

I further declare that I have discussed this use of artificial intelligence with my supervisor and received permission to use it.

Limerick, 2026

#heading(level: 1, numbering: none)[Ethics Declaration]

I herewith declare that my project does not involve human participants in any way and that I therefore was not required to submit an ethics application.

Limerick, 2026

pagebreak()

== Background and Context

Intrapartum CTG is central to fetal surveillance in labour, yet interpretation is still affected by inter-observer variation and uncertainty in borderline traces @figo2015_intro.

== Problem Statement

The core problem is to build a reliable and clinically useful classifier for fetal status from CTG, while reducing label noise and avoiding data leakage.

== Research Objectives

- Build an end-to-end CTG processing and modelling pipeline.
- Benchmark interpretable and deep-learning classifiers.
- Evaluate temporal alerting behaviour for real-time support.

== Research Questions

- How well can machine learning classify Normal, Suspicious, and Pathological CTG windows?
- Does a CNN on raw windows outperform a feature-based RF baseline?
- Can temporal smoothing improve alert stability without unacceptable performance loss?

== Project Scope and Limitations

This study focuses on the CTU-UHB dataset and weakly labelled windows. It does not include prospective clinical deployment.

== Expected Contributions

The dissertation contributes a reproducible benchmarking pipeline, leakage-aware RF baseline, CNN comparison, and post-processing analysis for clinician-facing alerts.

== Dissertation Outline

Chapter 2 reviews related work, Chapter 3 details methodology, Chapter 4 presents results, Chapter 5 discusses implications, and Chapter 6 concludes with future work.

= Literature Review and Background

== Cardiotocography: Clinical Context

CTG combines fetal heart rate and uterine activity monitoring to support intrapartum decisions @figo2015_intro @nwhip_guideline.

== Signal Processing for CTG Analysis

Robust CTG analysis depends on artifact handling, interpolation, smoothing, and physiologically bounded preprocessing @clifford2014_fecg.

== Machine Learning in Fetal Monitoring

Prior studies have used RF, SVM, boosting, and deep networks for CTG classification @hoodbhoy2019_ml @nagendra2017_realtime @innab2024_lgbm @zhang2024_svmcnn.

== Benchmarking and Performance Evaluation

Benchmarking requires controlled splits, class-aware metrics, and consistent reporting across models @khare2022_compare @kong2025_ensemble @m2025_icitiit.

== Trustworthy AI and Clinical Interpretability

Interpretability methods such as SHAP and local explanations are important for trustworthy clinical AI workflows @feng2023_shap @ribeiro2016_lime.

== Related Work and Research Gap

Recent reviews show improved model performance but highlight limited external validation and uncertainty in cross-database generalisation @francis2024_scoping @mendis2025_crossdb.

= Methodology

== Research Design

The project uses an experimental benchmarking design comparing RF and CNN under consistent preprocessing and split policies.

== Datasets

Primary experiments use CTU-UHB intrapartum recordings @chudacek2014_open @chudacek2014_ctuuhb @spilka2013_autoeval.

== Signal Preprocessing Pipeline

Preprocessing includes range checks, interpolation for short gaps, smoothing, and per-window normalization.

== Feature Extraction

Extracted domains include baseline, variability, acceleration/deceleration, spectral, entropy, and uterine contraction features.

== Machine Learning Classifiers

Two model families are benchmarked: Random Forest on engineered features and 1D-CNN on raw FHR+UC windows @fergus2020_cnn @cao2023_fusion @chiou2025_npj @pardasani2025_novelai.

== Training and Validation Strategy

Record-level grouped splitting is used to avoid leakage across windows from the same recording.

== Evaluation Metrics

Accuracy, macro-F1, weighted-F1, balanced accuracy, confusion matrices, and class-wise precision/recall are reported.

== Interpretability Analysis

Feature importance and domain-level attribution are used to explain model behaviour.

== Implementation Details

Implementation is in Python with reproducible notebooks, fixed random seeds, and exported reports.

= Results

== Data Characteristics

The final modelling set contains 29,989 windows across 552 recordings, with class imbalance toward Pathological windows.

== Signal Quality Analysis

Quality checks were applied before segmentation, including dropout handling and physiological bounds filtering.

== Feature Extraction Results

Feature extraction produced 35 leakage-aware model features used by the RF baseline.

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

== Statistical Analysis

Model comparison emphasizes class-wise behaviour, especially Pathological recall and confusion structure.

== Interpretability Analysis

RF feature importance indicates high influence from contraction and deceleration-related features.

== Visualisation

#figure(
  image("../outputs/models/rf_3class_confusion_matrix.png", width: 80%),
  caption: [Random Forest 3-class confusion matrix on held-out test windows.],
)<fig:rf_cm>

= Discussion

== Interpretation of Results

The RF model substantially outperforms the tested CNN configuration in this dataset setup.

== Benchmarking Insights

Leakage-aware feature engineering and grouped splitting are critical to realistic benchmark estimates.

== Clinical Implications

Temporal smoothing reduces alert transitions and can improve usability for clinical decision support.

== Comparison with State-of-the-Art

The findings are consistent with literature showing strong performance of interpretable tabular models in limited-data clinical settings @francis2024_scoping.

== Trustworthy AI Considerations

Clinical use requires transparency, threshold governance, and human-in-the-loop interpretation.

== Limitations

Key limitations include weak labels, class imbalance, and single-cohort training.

== Threats to Validity

Generalisability risk and potential distribution shift remain the primary threats to external validity.

= Conclusion and Future Work

== Summary of Findings

The developed pipeline provides reproducible CTG benchmarking and strong RF baseline performance.

== Contributions

Contributions include full workflow implementation, leakage controls, model benchmarking, and alert post-processing analysis.

== Practical Recommendations

Use grouped splits, report class-wise metrics, and tune thresholds to clinical objectives before deployment.

== Future Research Directions

Future work should include external cohorts, multimodal data fusion, and prospective validation studies.

== Closing Statement

This project supports the case for interpretable ML as a practical pathway to safer and more consistent CTG decision support.

= Appendix A

Supplementary implementation notes and configuration settings.

= Appendix B

Extended preprocessing diagnostics and quality summaries.

= Appendix C

Additional feature definitions and extraction details.

= Appendix D

Extended model outputs and error analysis tables.

= Appendix E

Additional figures for training dynamics and confusion matrices.

= Appendix F

Reproducibility checklist and run instructions.

#hide[bibliography("refs.bib")]

= Bibliography

Ayres-de-Campos, D., Arulkumaran, S., and Panel, F.I.F.M.E.C. (2015) 'FIGO consensus guidelines on intrapartum fetal monitoring: Introduction', International Journal of Gynecology \& Obstetrics, 131(1), 3-4, available: https://doi.org/10.1016/j.ijgo.2015.06.017.

Cao, Z., Wang, G., Xu, L., Li, C., Hao, Y., Chen, Q., Li, X., Liu, G., and Wei, H. (2023) 'Intelligent antepartum fetal monitoring via deep learning and fusion of cardiotocographic signals and clinical data', Health Information Science and Systems, 11(1), 16, available: https://doi.org/10.1007/s13755-023-00219-w.

Chiou, N., Young-Lin, N., Kelly, C., Cattiau, J., Tiyasirichokchai, T., Diack, A., Koyejo, S., Heller, K., and Asiedu, M. (2025) 'Development and evaluation of deep learning models for cardiotocography interpretation', npj Women's Health, 3(1), 21, available: https://doi.org/10.1038/s44294-025-00068-w.

Chudacek, V., Spilka, J., Bursa, M., Janku, P., Hruban, L., Huptych, M., and Lhotska, L. (2014a) 'Open access intrapartum CTG database', BMC Pregnancy and Childbirth, 14, 16, available: https://doi.org/10.1186/1471-2393-14-16.

Chudacek, V., Spilka, J., Bursa, M., Janku, P., Hruban, L., Huptych, M., and Lhotska, L. (2014b) 'Open access intrapartum CTG database', BMC Pregnancy and Childbirth, 14, 16, available: https://doi.org/10.1186/1471-2393-14-16.

Chudacek, V., Spilka, J., Bursa, M., Janku, P., Hruban, L., Huptych, M., and Lhotska, L. (2014c) 'The CTU-UHB Intrapartum Cardiotocography Database', available: https://doi.org/10.13026/C22013.

Clifford, G.D., Silva, I., Behar, J., and Moody, G.B. (2014) 'Noninvasive Fetal ECG analysis', Physiological Measurement, 35(8), 1521-1536, available: https://doi.org/10.1088/0967-3334/35/8/1521.

Feng, J., Liang, J., Qiang, Z., Hao, Y., Li, X., Li, L., Chen, Q., Liu, G., and Wei, H. (2023) 'A hybrid stacked ensemble and Kernel SHAP-based model for intelligent cardiotocography classification and interpretability', BMC Medical Informatics and Decision Making, 23, 273, available: https://doi.org/10.1186/s12911-023-02378-y.

Fergus, P., Chalmers, C., Montanez, C.C., Reilly, D., Lisboa, P., and Pineles, B. (2020) 'Modelling Segmented Cardiotocography Time-Series Signals Using One-Dimensional Convolutional Neural Networks for the Early Detection of Abnormal Birth Outcomes', available: https://doi.org/10.48550/arXiv.1908.02338.

Francis, F., Luz, S., Wu, H., Stock, S.J., and Townsend, R. (2024) 'Machine learning on cardiotocography data to classify fetal outcomes: A scoping review', Computers in Biology and Medicine, 172, 108220, available: https://doi.org/10.1016/j.compbiomed.2024.108220.

Hoodbhoy, Z., Noman, M., Shafique, A., Nasim, A., Chowdhury, D., and Hasan, B. (2019) 'Use of Machine Learning Algorithms for Prediction of Fetal Risk using Cardiotocographic Data', International Journal of Applied and Basic Medical Research, 9(4), 226-230, available: https://doi.org/10.4103/ijabmr.IJABMR_370_18.

Innab, N., Alsubai, S., Alabdulqader, E.A., Alarfaj, A.A., Umer, M., Trelova, S., and Ashraf, I. (2024) 'Automated approach for fetal and maternal health management using light gradient boosting model with SHAP explainable AI', Frontiers in Public Health, 12, available: https://doi.org/10.3389/fpubh.2024.1462693.

Khare, V. and Kumari, S. (2022) 'Performance Comparison of Three Classifiers for Fetal Health Classification Based on Cardiotocographic Data', Acadlore Transactions on AI and Machine Learning, 1(1), 52-60, available: https://doi.org/10.56578/ataiml010107.

Kong, L., Snasel, V., Bai, Z., Vilimek, D., Mirjalili, S., Pan, J.-S., Horakova, J., Martinek, R., and Vilimkova Kahankova, R. (2025) 'Enhancing cardiotocography classification via ensemble learning and threshold optimization', Scientific Reports, 15, 38528, available: https://doi.org/10.1038/s41598-025-18990-z.

M, H., Kumar, G.S., S, G., and Mishra, M. (2025) 'Comparative Study of Machine Learning Algorithms to Predict Fetal Health Using Cardiotocography Data', in 2025 International Conference on Innovative Trends in Information Technology (ICITIIT), Presented at the 2025 International Conference on Innovative Trends in Information Technology (ICITIIT), 1-6, available: https://doi.org/10.1109/ICITIIT64777.2025.11040542.

Mendis, L., Karmakar, D., Palaniswami, M., Brownfoot, F., and Keenan, E. (2025) 'Cross-Database Evaluation of Deep Learning Methods for Intrapartum Cardiotocography Classification', IEEE Journal of Translational Engineering in Health and Medicine, 13, 123-135, available: https://doi.org/10.1109/JTEHM.2025.3548401.

Modelling Segmented Cardiotocography Time-Series Signals Using One-Dimensional Convolutional Neural Networks for the Early Detection of Abnormal Birth Outcomes [online] (2026) ar5iv, available: https://ar5iv.labs.arxiv.org/html/1908.02338 [accessed 25 Mar 2026].

Nagendra, V., Gude, H., Sampath, D., Corns, S., and Long, S. (2017) 'Evaluation of support vector machines and random forest classifiers in a real-time fetal monitoring system based on cardiotocography data', in 2017 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB), Presented at the 2017 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB), 1-6, available: https://doi.org/10.1109/CIBCB.2017.8058546.

'National Clinical Practice Guideline Fetal Heart Rate Monitoring' (n.d.).

Nazli, I., Korbeko, E., Dogru, S., Kugu, E., and Sahingoz, O.K. (2025) 'Early Detection of Fetal Health Conditions Using Machine Learning for Classifying Imbalanced Cardiotocographic Data', Diagnostics, 15(10), 1250, available: https://doi.org/10.3390/diagnostics15101250.

Pardasani, R., Vitullo, R., Harris, S., Yapici, H.O., and Beard, J. (2025) 'Development of a novel artificial intelligence algorithm for interpreting fetal heart rate and uterine activity data in cardiotocography', Frontiers in Digital Health, 7, 1638424, available: https://doi.org/10.3389/fdgth.2025.1638424.

Ribeiro, M.T., Singh, S., and Guestrin, C. (2016) '"Why Should I Trust You?": Explaining the Predictions of Any Classifier', available: https://doi.org/10.48550/arXiv.1602.04938.

Spilka, J., Georgoulas, G., Karvelis, P., Oikonomou, V.P., Chudacek, V., Stylios, C., Lhotska, L., and Janku, P. (2013) 'Automatic Evaluation of FHR Recordings from CTU-UHB CTG Database', in Bursa, M., Khuri, S. and Renda, M.E., eds, Information Technology in Bio- and Medical Informatics, Lecture Notes in Computer Science, Berlin, Heidelberg: Springer Berlin Heidelberg, 47-61, available: https://doi.org/10.1007/978-3-642-40093-3_4.

Zhang, W., Tang, Z., Shao, H., Sun, C., He, X., Zhang, J., Wang, T., Yang, X., Wang, Y., Bin, Y., Zhao, L., Zhang, S., Liang, D., Wang, J., Zhong, D., and Li, Q. (2024) 'Intelligent classification of cardiotocography based on a support vector machine and convolutional neural network: Multiscene research', International Journal of Gynaecology and Obstetrics: The Official Organ of the International Federation of Gynaecology and Obstetrics, 165(2), 737-745, available: https://doi.org/10.1002/ijgo.15236.
