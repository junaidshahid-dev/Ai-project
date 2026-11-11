"""Build the Phase 5 report phase5/F26-11.pdf (numbers come from the executed notebook F26-11.ipynb and
from the saved results of the full experiment run)."""
from pathlib import Path

import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
OUT = HERE / "F26-11.pdf"
FIG = HERE / "figures"
REP = pd.read_csv(HERE / "reproducibility.csv")
NORM = pd.read_csv(HERE / "speaker_normalisation.csv", index_col=0)
SHIFT = pd.read_csv(HERE / "time_shift.csv", index_col=0)

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=14.5, leading=18, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, alignment=1, fontSize=10, textColor=colors.HexColor("#333333"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12, spaceBefore=6, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=9, leading=10.8, spaceAfter=0)
cap = ParagraphStyle("cap", parent=body, fontSize=9, leading=11, alignment=1, textColor=colors.HexColor("#333333"))
code = "<font name='Courier' size='9'>{}</font>".format


def pct(x):
    return f"{100 * x:.1f}"


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=12) for t in items],
                        bulletType="bullet", start="•", leftIndent=14)


def table(rows, widths, bold_last=False):
    t = Table([[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=[w * cm for w in widths])
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8"))]
    if bold_last:
        style.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f3f7ff")))
    t.setStyle(TableStyle(style))
    return t


def image(name, width_cm):
    w, h = Image.open(FIG / f"{name}.png").size
    return RLImage(str(FIG / f"{name}.png"), width=width_cm * cm, height=width_cm * cm * h / w)


def figure(name, width_cm, caption):
    return KeepTogether([image(name, width_cm), Paragraph(caption, cap), Spacer(1, 3)])


def link(u):
    return f'<link href="{u}" color="blue">{u}</link>'


plain, norm = NORM.loc["without"], NORM.loc["with speaker normalisation"]
story = [
    Paragraph("National University of Computer &amp; Emerging Sciences, Lahore", sub),
    Paragraph("AI2002 – Artificial Intelligence (BS-SE-5A) — Project Deliverable 5: Technical Report "
              "(Track B)", sub),
    Spacer(1, 4),
    Paragraph("Lightweight Speech Emotion Recognition Using Hand-Crafted Acoustic Features: "
              "Experimental Results, Theoretical Insights and Reproducibility", h1),
    Spacer(1, 2),
]
info = [
    [Paragraph("<b>Group ID</b>", cell), Paragraph("F26-11", cell),
     Paragraph("<b>Track</b>", cell), Paragraph("B – Research &amp; Development", cell)],
    [Paragraph("<b>Members</b>", cell), Paragraph("22L-7786 (Lead), 23L-3002, 23L-3102", cell),
     Paragraph("<b>Instructor</b>", cell), Paragraph("Hajra Waheed", cell)],
    [Paragraph("<b>Code</b>", cell), Paragraph("F26-11.ipynb (also F26-11.py) · "
               + link("https://github.com/junaidshahid-dev/Ai-project") + " (folder phase5/)", cell), "", ""],
]
t = Table(info, colWidths=[2.2 * cm, 6.4 * cm, 2.2 * cm, 6.0 * cm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("SPAN", (1, 2), (3, 2)),
                       ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
                       ("BACKGROUND", (2, 0), (2, 1), colors.HexColor("#eeeeee"))]))
story += [t, Spacer(1, 5)]

# 1. Experimental results
story += [
    Paragraph("1. Experimental Results", h2),
    Paragraph("We re-implemented the lightweight SER ensemble of Chowdhury et al. (<i>Sci. Rep.</i> 15, 11824, "
              "2025): 34 hand-crafted features per frame (ZCR, RMS energy, 20 MFCCs, 12 Chroma) for 108 frames, "
              "flattened to 3,672 values, fed to a 1D-CNN and a CNN-BiLSTM whose probabilities are averaged. The "
              "paper reports 97.57% on RAVDESS. Table 1 summarises our results (details in Deliverable 4).", body),
    table([
        ["Protocol", "Test data", "Ensemble", "SVM", "Paper"],
        ["Augment before split (leaky)", "576 RAVDESS samples", "96.5%", "85.6%", "97.57%"],
        ["Clean random split (80/10/10)", "144 RAVDESS clips", "81.9%", "77.1%", "–"],
        ["Unseen speakers (4-fold)", "1,440 clips, new actors", "58.5% ± 1.9", "49.1%", "–"],
        ["RAVDESS → TESS", "2,800 TESS clips", "19.7%", "23.9%", "– (chance 14.3%)"],
        ["Gender gap, unseen speakers", "720 clips per gender", "F 66.2% / M 50.7%", "F 52.9% / M 45.3%", "–"],
    ], [5.0, 3.8, 2.9, 2.9, 2.2]),
    Paragraph("Table 1. Accuracy of the paper's ensemble and the SVM baseline under each protocol.", cap),
    Paragraph("<b>Error analysis in brief.</b> Most errors are between acoustically similar emotions (sad ↔ "
              "neutral, happy → fearful, angry → disgust); sad is the hardest class for new speakers (34.4% "
              "recall); strong intensity is far easier than normal intensity (66.1% vs 47.9%); accuracy per actor "
              "ranges from 40.0% to 81.7%; and on TESS the model predicts <i>sad</i> for 53.1% of clips, largely "
              "because 41.5% of TESS frames are zero padding. Below we test four explanations with new "
              "experiments, using the SVM because it trains in seconds.", body),
]

# 2. Reproducibility
story += [
    Paragraph("2. Reproducibility of the Analysis", h2),
    bullets([
        "<b>Every reported number is recomputed.</b> The notebook recomputes all 30 accuracies of the experiment "
        "summary (6 models × 5 protocols) from the saved per-clip predictions: all are identical.",
        f"<b>Re-training gives the same predictions.</b> Re-training the SVM from the feature cache with seed 42 "
        f"on this laptop gives the <b>same prediction for {pct(REP.loc[0, 'same predictions'])}%</b> of the random-"
        f"split test clips ({pct(REP.loc[0, 'new accuracy'])}% accuracy, as saved) and "
        f"{pct(REP.loc[2, 'same predictions'])}% of the 1,440 unseen-speaker clips ({pct(REP.loc[2, 'new accuracy'])}% "
        f"vs {pct(REP.loc[2, 'saved accuracy'])}% saved). The Random Forest agrees on "
        f"{pct(REP.loc[1, 'same predictions'])}% ({pct(REP.loc[1, 'new accuracy'])}% vs "
        f"{pct(REP.loc[1, 'saved accuracy'])}%): its trees depend on the library version and the number of CPU "
        "threads, and deep models change in the same way (our Phase 3 re-run gave the ensemble 77.1%).",
        "<b>Single test splits are noisy (Figure 3, left).</b> Repeating the random 80/10/10 split with 10 seeds "
        "gives the same SVM between <b>72.9% and 83.3%</b> (mean 76.9%, std 4.1 points). Theory predicts a "
        "standard error of √(p(1−p)/n) = 3.5 points for n = 144 test clips. So a difference of a few points "
        "between two models, or between our run and a re-run, is within normal variation; conclusions should "
        "rest on large test sets such as the 1,440-clip speaker-independent protocol.",
    ]),
]

# 3. Theoretical insights
story += [
    Paragraph("3. Theoretical Insights and Follow-up Experiments", h2),
    Paragraph("<b>Insight 1 – the features describe the speaker more than the emotion.</b> For each of the 68 "
              "summary features (mean and std of the 34 features over time) we computed eta squared, the share of "
              "its variance explained by group means. On average the speaker explains 27.9% of a feature's "
              "variance and the emotion only 7.6%; the speaker dominates in 65 of the 68 features. The same "
              "SVM recognises <b>who is speaking with 95.0% accuracy</b> (24 actors, chance 4.2%) but the "
              "emotion with 74.4% (5-fold CV). Only RMS energy is mainly driven by emotion (34.2% vs 4.0%). "
              "This explains the drop for unseen speakers: a classifier can reach high within-speaker accuracy "
              "by learning each speaker's typical feature values, and those values are wrong for a new voice.",
              body),
    figure("fig1_speaker_vs_emotion", 11.5, "Figure 1. Average share of feature variance explained by the speaker "
           "and by the emotion (RAVDESS, 1,440 original clips)."),
    Paragraph("<b>Insight 2 – removing speaker identity helps, but does not remove the gender gap.</b> If "
              "speaker identity is the problem, z-scoring each speaker's features with that speaker's own mean "
              "and std should help. This needs no labels, so it is also possible for a new user (a short "
              "enrolment recording). Results with the SVM (Figure 2):", body),
    bullets([
        f"Unseen speakers: <b>{pct(plain['unseen speakers'])}% → {pct(norm['unseen speakers'])}%</b> "
        f"(+{100 * (norm['unseen speakers'] - plain['unseen speakers']):.1f} points). The simple SVM then matches "
        "the paper's deep ensemble without normalisation (58.5%).",
        f"RAVDESS → TESS: <b>{pct(plain['RAVDESS -> TESS'])}% → {pct(norm['RAVDESS -> TESS'])}%</b>, the best "
        "cross-dataset result in the project, because normalisation also removes the differences in recording "
        "level and voice between the two corpora.",
        f"Gender gap: {100 * plain['gap (F - M)']:.1f} → <b>{100 * norm['gap (F - M)']:.1f} points</b>. Both genders "
        f"improve (male {pct(plain['male'])}% → {pct(norm['male'])}%, female {pct(plain['female'])}% → "
        f"{pct(norm['female'])}%), but women gain more. The bias is therefore <i>not</i> caused by different "
        "voice baselines; the emotional changes in the male voices of RAVDESS are harder to separate with these "
        "features (possibly because of a smaller pitch range, which Chroma and MFCC capture). Fixing it would need "
        "other features or gender-balanced model selection.",
    ]),
    figure("fig2_speaker_normalisation", 11.5, "Figure 2. SVM accuracy with and without per-speaker "
           "normalisation."),
    Paragraph("<b>Insight 3 – the flattened input is tied to time positions.</b> The paper's 3,672-value vector "
              "gives each frame its own position, so the same speech starting a little later produces a different "
              "input. We trained an SVM on the flattened input and one on the 68-value summary, and tested both "
              f"on test clips delayed by 0–40 frames (Figure 3, right). A delay of only 0.12 s (5 frames) costs "
              f"the flattened input {100 * (SHIFT.iloc[0, 0] - SHIFT.iloc[1, 0]):.1f} points "
              f"({pct(SHIFT.iloc[0, 0])}% → {pct(SHIFT.iloc[1, 0])}%) but the summary only "
              f"{100 * (SHIFT.iloc[0, 1] - SHIFT.iloc[1, 1]):.1f} ({pct(SHIFT.iloc[0, 1])}% → "
              f"{pct(SHIFT.iloc[1, 1])}%); at 0.23 s the losses are {100 * (SHIFT.iloc[0, 0] - SHIFT.iloc[2, 0]):.1f} "
              f"and {100 * (SHIFT.iloc[0, 1] - SHIFT.iloc[2, 1]):.1f} points. This explains why the Random Forest, which "
              "uses the summary, transfers best to TESS (31.1%), and why the zero padding of short TESS clips "
              "hurts the deep models: loading TESS from 0 s instead of 0.6 s raised the ensemble from 19.7% to "
              "30.5%. Convolutions detect a pattern wherever it occurs, but the Flatten and Dense layers after "
              "them again assign weights to fixed positions.", body),
    figure("fig3_split_and_shift", 15.5, "Figure 3. Left: SVM accuracy on 10 random splits (dashed: the split used in "
           "the project). Right: accuracy when the test speech is delayed."),
    Paragraph("<b>Insight 4 – why augmentation before splitting inflates accuracy.</b> A noisy or pitch-shifted "
              "copy of a recording is almost the same input as the original. When copies are split at random, "
              "99.1% of test samples have a copy in training, so a high-capacity model can recognise the recording "
              "instead of the emotion. The test set is then no longer independent of the training set, which every "
              "accuracy estimate assumes. This is the only setting in which we reach the paper's level (96.5%).",
              body),
]

story += [
    KeepTogether([Paragraph("4. Conclusions", h2), bullets([
        "The paper's architecture works (81.9% on a clean split, best model for unseen speakers), but its 97.57% "
        "is only reproduced with leakage between training and test data.",
        "The main weakness is speaker dependence: the features carry more information about the speaker than "
        "about the emotion. Per-speaker normalisation is a cheap, label-free fix (+9.9 points for new speakers, "
        "+10.6 on TESS with an SVM) and should be applied to the deep models next.",
        "The gender gap is a separate problem that normalisation does not solve; it must be reported and "
        "checked before any deployment.",
        "Results must come with their uncertainty: on 144 test clips, the split alone moves accuracy by up to 10 "
        "points.",
    ])]),
    KeepTogether([Paragraph("5. Challenges Faced", h2), bullets([
        "<b>Testing explanations cheaply.</b> Re-training the deep models takes hours on our 2-core CPU, so the "
        "follow-up experiments use the SVM on the same features. It follows the same trends as the deep models "
        "(good on a random split, much weaker for new speakers, near chance on TESS), but the effect of normalisation on the deep ensemble itself is still to be measured.",
        "<b>Exact reproducibility.</b> Seeds alone are not enough: Random Forest and neural-network results "
        "change with the library version and number of CPU threads. We save all predictions and report how many "
        "of them a re-run reproduces.",
        "<b>A surprising result.</b> We expected speaker normalisation to reduce the gender gap; it increased it. "
        "We report it as it is, because it shows that the bias has a different cause.",
    ])]),
    Paragraph("<b>To reproduce:</b> run " + code("phase5/F26-11.ipynb") + " (about 6 minutes on a laptop CPU; "
              "needs the Phase 2 feature cache, no TensorFlow). All seeds are 42.", body),
]

SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm,
                  bottomMargin=1.5 * cm, title="F26-11 Deliverable 5 – Technical Report",
                  author="F26-11 (22L-7786, 23L-3002, 23L-3102)").build(story)
print("wrote", OUT)
