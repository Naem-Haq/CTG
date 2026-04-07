from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt


ROOT = Path('/home/naem-haq/Software-Engineering/CTG')
OUT = ROOT / 'docs' / 'ctg_benchmark_presentation.pptx'
IMAGES = {
    'rf_cm': ROOT / 'outputs/models/rf_3class_confusion_matrix.png',
    'cnn_cm': ROOT / 'outputs/models/cnn_3class_confusion_matrix.png',
    'rf_fi': ROOT / 'outputs/models/rf_3class_feature_importance.png',
    'alerts': ROOT / 'outputs/models/blockF_alert_timeline.png',
}


def set_title(slide, title):
    slide.shapes.title.text = title
    slide.shapes.title.text_frame.paragraphs[0].font.size = Pt(34)


def add_bullets(prs, title, bullets, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    set_title(slide, title)
    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()

    if subtitle:
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.bold = True
        p.font.size = Pt(20)

    for i, b in enumerate(bullets):
        p = tf.add_paragraph() if (subtitle or i > 0) else tf.paragraphs[0]
        p.text = b
        p.level = 0
        p.font.size = Pt(20)

    return slide


def add_two_images(prs, title, left_path, right_path, caption_left, caption_right):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(slide, title)

    left_x, right_x = Inches(0.5), Inches(5.1)
    top_img = Inches(1.3)
    max_w, max_h = Inches(4.4), Inches(4.7)

    def place(path, x):
        with Image.open(path) as im:
            w, h = im.size
        ratio = min(max_w / w, max_h / h)
        disp_w = w * ratio
        disp_h = h * ratio
        y = top_img + (max_h - disp_h) / 2
        slide.shapes.add_picture(str(path), x, y, width=disp_w, height=disp_h)

    place(left_path, left_x)
    place(right_path, right_x)

    tb1 = slide.shapes.add_textbox(left_x, Inches(6.2), max_w, Inches(0.5)).text_frame
    tb1.text = caption_left
    tb1.paragraphs[0].font.size = Pt(14)

    tb2 = slide.shapes.add_textbox(right_x, Inches(6.2), max_w, Inches(0.5)).text_frame
    tb2.text = caption_right
    tb2.paragraphs[0].font.size = Pt(14)

    return slide


def set_notes(slide, text):
    notes = slide.notes_slide.notes_text_frame
    notes.clear()
    notes.text = text


def build_presentation():
    prs = Presentation()

    cover = prs.slides.add_slide(prs.slide_layouts[0])
    cover.shapes.title.text = 'Benchmarking ML Classifiers for CTG\nFetal Heart Rate Pattern Detection'
    cover.shapes.title.text_frame.paragraphs[0].font.size = Pt(36)
    cover.placeholders[1].text = (
        'Naem Haq\n'
        'FYP Dissertation Presentation\n'
        'Residency 4 context: INFANT Research Centre (UCC)'
    )
    for p in cover.placeholders[1].text_frame.paragraphs:
        p.font.size = Pt(18)
    set_notes(
        cover,
        'Good [morning/afternoon]. I am Naem Haq, and this presentation summarizes my FYP benchmarking '
        'study on machine-learning classifiers for CTG fetal heart rate pattern detection. The work is framed '
        'by my Residency 4 context at INFANT Research Centre, and focuses on fair, reproducible model '
        'comparison for decision-support relevance.',
    )

    s2 = add_bullets(
        prs,
        'Motivation and Context',
        [
            'CTG is widely used in labour monitoring, but interpretation can be subjective.',
            'Clinical variability motivates transparent decision-support benchmarking.',
            'This benchmarking focus was shaped during Residency 4 at INFANT Research Centre.',
        ],
    )
    set_notes(
        s2,
        'CTG is a core intrapartum monitoring tool, but interpretation can be subjective and variable across '
        'clinicians and settings. That variability motivates computational decision support. During my Residency '
        '4 placement at INFANT, this practical need became clear, which directly motivated a controlled '
        'benchmarking study rather than an isolated model-building exercise.',
    )

    s3 = add_bullets(
        prs,
        'Aim and Scope',
        [
            'Aim: benchmark supervised classifiers for fetal heart rate pattern detection.',
            'Models compared: Random Forest (feature-based) vs 1D-CNN (raw-signal).',
            'Scope: within-cohort CTU-UHB benchmark with weakly supervised labels.',
            'Claim boundary: fixed-split comparative evidence, not inferential repeated-run ranking.',
        ],
    )
    set_notes(
        s3,
        'The aim was to benchmark two model families under the same conditions: Random Forest and 1D-CNN. '
        'This is a within-cohort CTU-UHB benchmark with weakly supervised labels. I keep claims bounded to '
        'fixed-split comparative evidence, so conclusions are transparent and proportional to executed analysis.',
    )

    s4 = add_bullets(
        prs,
        'Benchmark Design and Fairness Controls',
        [
            'Shared preprocessing and leakage-aware grouped splitting by record ID.',
            'Common evaluation criteria: class-wise and aggregate metrics.',
            'Additional evidence: efficiency, confusion analysis, threshold behaviour, interpretability.',
            'Reproducibility controls: fixed seed policy, split manifests, JSON/CSV artifacts.',
        ],
    )
    set_notes(
        s4,
        'A key contribution is fairness of comparison. Both models share the same preprocessing, grouped '
        'splitting, and evaluation framework. Record-level grouping prevents leakage between train and test '
        'windows from the same labour record. Reproducibility is supported by fixed seeds, stored split '
        'manifests, and exported artifacts for auditability.',
    )

    s5 = add_bullets(
        prs,
        'Pipeline',
        [
            'Signal quality control and preprocessing',
            'Windowing and feature extraction',
            'Model training with grouped holdout protocol',
            'Evaluation, error analysis, thresholding, and post-processing review',
        ],
    )
    set_notes(
        s5,
        'The pipeline runs from raw CTG through quality control, preprocessing, segmentation and feature '
        'construction, then model training and evaluation. Beyond headline metrics, I include class-wise error '
        'analysis, threshold behavior, and post-processing effects so the benchmark reflects operational '
        'decision-support needs.',
    )

    s6 = add_bullets(
        prs,
        'Headline Results',
        [
            'Held-out windows: RF 95.55% accuracy, macro-F1 0.945.',
            'Held-out windows: CNN 65.31% accuracy, macro-F1 0.584.',
            'Record-level CNN drops to 48.19% accuracy, macro-F1 0.383.',
            'Result: RF is the stronger current baseline under this data regime.',
        ],
    )
    set_notes(
        s6,
        'The performance gap is substantial on held-out windows: RF reaches 95.55% accuracy and macro-F1 '
        '0.945, while CNN reaches 65.31% and 0.584. At record level, CNN drops further to 48.19% accuracy '
        'and macro-F1 0.383. Under this tested regime, Random Forest is the stronger baseline.',
    )

    if IMAGES['rf_cm'].exists() and IMAGES['cnn_cm'].exists():
        s7 = add_two_images(
            prs,
            'Confusion Matrix Comparison',
            IMAGES['rf_cm'],
            IMAGES['cnn_cm'],
            'Random Forest: tighter class separation',
            '1D-CNN: greater off-diagonal confusion',
        )
        set_notes(
            s7,
            'These confusion matrices show how the aggregate metric gap appears structurally. RF displays '
            'stronger diagonal concentration, indicating better class separation. The CNN matrix shows broader '
            'off-diagonal confusion, especially around boundary classes, which contributes to weaker macro-F1 '
            'and lower stability after aggregation.',
        )

    if IMAGES['rf_fi'].exists() and IMAGES['alerts'].exists():
        s8 = add_two_images(
            prs,
            'Interpretability and Alert Dynamics',
            IMAGES['rf_fi'],
            IMAGES['alerts'],
            'RF feature/domain importance',
            'Temporal alert timeline after smoothing',
        )
        set_notes(
            s8,
            'On the left, RF importance outputs provide interpretable evidence about which domains drive model '
            'decisions, supporting transparency. On the right, the alert timeline illustrates temporal behavior '
            'under post-processing policy, showing that usability and alert stability depend on threshold and '
            'smoothing choices, not only classifier architecture.',
        )

    s9 = add_bullets(
        prs,
        'Conclusion',
        [
            'Under tested conditions, Random Forest is the more suitable transparent baseline.',
            'This does not imply deep learning is always worse in CTG.',
            'The contribution is an auditable benchmark reference for next-stage validation.',
        ],
    )
    set_notes(
        s9,
        'The conclusion is conditional and scoped: under the tested conditions, Random Forest is currently '
        'the more suitable transparent benchmark baseline. This is not a universal claim against deep learning. '
        'The study contributes an auditable foundation that future work can extend.',
    )

    s10 = add_bullets(
        prs,
        'Limitations and Next Steps',
        [
            'Findings are bounded to CTU-UHB and weak-label regime.',
            'Repeated grouped resampling and stronger uncertainty quantification are future extensions.',
            'External validation across broader datasets is required for stronger transportability claims.',
            'Pathway: prospective, clinician-facing translation aligned with INFANT/AI4Life goals.',
        ],
    )
    set_notes(
        s10,
        'These findings are specific to the CTU-UHB cohort and weak-label setting used in this benchmark. Next steps are uncertainty-focused repeat evaluation and external validation on broader datasets to support prospective clinician-facing translation in INFANT/AI4Life contexts.'
    )

    prs.save(OUT)
    print(OUT)


if __name__ == '__main__':
    build_presentation()
