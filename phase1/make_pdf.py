from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem

OUT = "/home/user/Ai-project/phase1/F26-GroupID.pdf"

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=11, leading=14.5, spaceAfter=4)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=15, leading=19, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, alignment=1, fontSize=10.5, textColor=colors.HexColor("#333333"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12.5, spaceBefore=8, spaceAfter=4)
cell = ParagraphStyle("c", parent=body, fontSize=10.5, leading=13, spaceAfter=0)

def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=12) for t in items],
                        bulletType="bullet", start="•", leftIndent=14)

def link(u):
    return f'<link href="{u}" color="blue">{u}</link>'

story = [
    Paragraph("National University of Computer &amp; Emerging Sciences, Lahore", sub),
    Paragraph("AI2002 – Artificial Intelligence (BS-SE-5A) — Project Phase 1", sub),
    Spacer(1, 6),
    Paragraph("Lightweight Speech Emotion Recognition Using Hand-Crafted Acoustic Features: "
              "Cross-Dataset Generalization and Gender-Bias Analysis", h1),
    Spacer(1, 4),
]

info = [
    [Paragraph("<b>Group ID</b>", cell), Paragraph("F26-____", cell),
     Paragraph("<b>Track</b>", cell), Paragraph("B – Research &amp; Development", cell)],
    [Paragraph("<b>Members</b>", cell), Paragraph("22L-7786 (Lead)<br/>23L-3002", cell),
     Paragraph("<b>Instructor</b>", cell), Paragraph("Hajra Waheed", cell)],
]
t = Table(info, colWidths=[2.3*cm, 5.6*cm, 2.3*cm, 6.2*cm])
t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                       ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eeeeee")),
                       ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#eeeeee")),
                       ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
story += [t]

story += [
    Paragraph("1. Problem Statement", h2),
    Paragraph(
        "Speech Emotion Recognition (SER) classifies a speaker's emotional state (e.g., angry, happy, sad, "
        "neutral) from voice alone, with uses in call-centre analytics, mental-health screening and voice "
        "assistants. Chowdhury et al. (2025) show that a lightweight ensemble (CNN + CNN-BiLSTM) trained on "
        "hand-crafted acoustic features reaches high accuracy while remaining far cheaper than large "
        "pretrained models. However, such models are typically evaluated by training and testing on the "
        "<i>same</i> dataset, which hides two practical risks:", body),
    bullets([
        "<b>Generalization:</b> performance may collapse on unseen speakers and recording conditions.",
        "<b>Bias:</b> accuracy may differ systematically between male and female speakers.",
    ]),
    Paragraph(
        "<b>Hypothesis:</b> a lightweight hand-crafted-feature SER model that performs well within a dataset "
        "will lose significant accuracy in a cross-dataset setting and show a measurable gender performance gap. "
        "This project tests that hypothesis and identifies which acoustic features matter most.", body),

    Paragraph("2. Dataset", h2),
]

ds = [
    [Paragraph("<b>Dataset</b>", cell), Paragraph("<b>Type</b>", cell), Paragraph("<b>Details</b>", cell), Paragraph("<b>Role</b>", cell)],
    [Paragraph("RAVDESS<br/>(Kaggle)", cell), Paragraph("Audio (.wav)", cell),
     Paragraph("1,440 clips, 24 actors (12 M / 12 F), 8 emotions", cell), Paragraph("Training, within-dataset testing, gender-bias check", cell)],
    [Paragraph("TESS<br/>(Kaggle)", cell), Paragraph("Audio (.wav)", cell),
     Paragraph("7 emotions, 2 female speakers", cell), Paragraph("Unseen cross-dataset test set", cell)],
]
t2 = Table(ds, colWidths=[2.6*cm, 2.5*cm, 5.6*cm, 5.7*cm])
t2.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeeee")),
                        ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
story += [t2, Spacer(1, 4),
    Paragraph("RAVDESS: " + link("https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio"), body),
    Paragraph("TESS: " + link("https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"), body),

    Paragraph("3. Base Paper", h2),
    Paragraph("J. H. Chowdhury, S. Ramanna, K. Kotecha, “Speech emotion recognition with light weight deep "
              "neural ensemble model using hand crafted features,” <i>Scientific Reports</i>, vol. 15, "
              "art. 11824, 2025. DOI: 10.1038/s41598-025-95734-z", body),
    Paragraph("Link: " + link("https://www.nature.com/articles/s41598-025-95734-z"), body),

    Paragraph("4. Scope of the Project", h2),
    Paragraph("<b>In scope:</b>", body),
    bullets([
        "<b>Preprocessing:</b> silence trimming, normalization, augmentation (noise, pitch shift, time stretch); "
        "mapping both datasets to a common emotion label set.",
        "<b>Feature extraction:</b> MFCC, Chroma STFT, Zero-Crossing Rate, RMS energy (librosa).",
        "<b>Models:</b> baselines (SVM, Random Forest) vs. 1D-CNN, CNN-BiLSTM and their ensemble (following the paper), "
        "using Scikit-Learn and TensorFlow/PyTorch.",
        "<b>Experiments:</b> (i) within-dataset evaluation on RAVDESS; (ii) cross-dataset evaluation "
        "(train RAVDESS → test TESS); (iii) feature ablation, removing one feature group at a time.",
        "<b>Evaluation:</b> accuracy, macro-F1, per-class confusion matrices, model size and inference time.",
        "<b>Bias check:</b> male vs. female accuracy and F1 on RAVDESS.",
        "<b>Deliverables:</b> reproducible codebase and a LaTeX technical report with results and error analysis.",
    ]),
    Paragraph("<b>Out of scope:</b> large pretrained speech models (e.g., wav2vec 2.0, LLMs), real-time "
              "deployment, non-English or conversational (non-acted) speech. A simple Streamlit demo may be "
              "added only as a bonus.", body),
]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="Phase 1 - Speech Emotion Recognition", author="22L-7786, 23L-3002")
doc.build(story)
print(OUT)
