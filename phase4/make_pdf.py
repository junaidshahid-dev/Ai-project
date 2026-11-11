"""Build the Phase 4 report phase4/F26-11.pdf (numbers come from the executed notebook F26-11.ipynb)."""
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
M = pd.read_csv(HERE / "metrics.csv")
MC = pd.read_csv(HERE / "mcnemar.csv")
BIAS = pd.read_csv(HERE / "gender_bias.csv")

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=14.5, leading=18, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, alignment=1, fontSize=10, textColor=colors.HexColor("#333333"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12, spaceBefore=6, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=9, leading=10.8, spaceAfter=0)
cap = ParagraphStyle("cap", parent=body, fontSize=9, leading=11, alignment=1, textColor=colors.HexColor("#333333"))
code = "<font name='Courier' size='9'>{}</font>".format
NAMES = {"SVM": "SVM", "RandomForest": "Random Forest", "CNN": "1D-CNN", "CNN_BiLSTM": "CNN-BiLSTM",
         "Ensemble": "Ensemble", "Majority": "Majority (chance)"}
EXPS = ["paper", "leaky", "speaker", "ravdess_to_tess", "tess_to_ravdess"]


def row(exp, model):
    return M[(M["experiment"] == exp) & (M["model"] == model)].iloc[0]


def acc_ci(exp, model):
    r = row(exp, model)
    return f"{100 * r['accuracy']:.1f} [{100 * r['ci_low']:.0f}–{100 * r['ci_high']:.0f}]"


def pct(exp, model):
    return f"{100 * row(exp, model)['accuracy']:.1f}%"


def mcn(protocol, comparison):
    r = MC[(MC["protocol"] == protocol) & (MC["comparison"] == comparison)].iloc[0]
    p = "&lt; 0.001" if r["p_value"] < 0.001 else f"= {r['p_value']:.2f}"
    return f"{r['only A right']} vs {r['only B right']} clips, p {p}"


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


story = [
    Paragraph("National University of Computer &amp; Emerging Sciences, Lahore", sub),
    Paragraph("AI2002 – Artificial Intelligence (BS-SE-5A) — Project Deliverable 4: Evaluation and Analysis of "
              "Results", sub),
    Spacer(1, 4),
    Paragraph("Lightweight Speech Emotion Recognition Using Hand-Crafted Acoustic Features: "
              "Cross-Dataset Generalization and Gender-Bias Analysis", h1),
    Spacer(1, 2),
]
info = [
    [Paragraph("<b>Group ID</b>", cell), Paragraph("F26-11", cell),
     Paragraph("<b>Track</b>", cell), Paragraph("B – Research &amp; Development", cell)],
    [Paragraph("<b>Members</b>", cell), Paragraph("22L-7786 (Lead), 23L-3002, 23L-3102", cell),
     Paragraph("<b>Instructor</b>", cell), Paragraph("Hajra Waheed", cell)],
    [Paragraph("<b>Code</b>", cell), Paragraph("F26-11.ipynb (also F26-11.py) · "
               + link("https://github.com/junaidshahid-dev/Ai-project") + " (folder phase4/)", cell), "", ""],
]
t = Table(info, colWidths=[2.2 * cm, 6.4 * cm, 2.2 * cm, 6.0 * cm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("SPAN", (1, 2), (3, 2)),
                       ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
                       ("BACKGROUND", (2, 0), (2, 1), colors.HexColor("#eeeeee"))]))
story += [t, Spacer(1, 5)]

# 1. Setup
story += [
    Paragraph("1. Evaluation Setup", h2),
    Paragraph("We evaluate the five models of Phase 3 (SVM, Random Forest, the paper's 1D-CNN and CNN-BiLSTM, and "
              "their averaging ensemble) on 7-class speech emotion recognition. Each model is trained and tested "
              "under five protocols, from the base paper's own setting to the strictest one (no speaker or dataset "
              "overlap). All per-clip predictions are saved by " + code("scripts/run_experiments.py") + " (the same "
              "model code as Phase 3); the notebook " + code("phase4/F26-11.ipynb") + " computes every number below "
              "from them in seconds and checks them against " + code("results/summary.json") + ".", body),
    table([
        ["Protocol", "Training data", "Test data", "Question"],
        ["P1 Random split (paper)", "80% of RAVDESS clips + their augmented copies", "144 RAVDESS clips",
         "Does the model work?"],
        ["P2 Augment before split", "80% of the 5,760 augmented samples", "576 samples",
         "Does the split order inflate accuracy?"],
        ["P3 Unseen speakers", "18 actors (4 folds, 3 M + 3 F held out)", "all 1,440 clips, 6 new actors per fold",
         "Does it work for new voices?"],
        ["P4 RAVDESS → TESS", "all RAVDESS", "2,800 TESS clips", "Does it work on a new dataset?"],
        ["P5 TESS → RAVDESS", "all TESS", "1,440 RAVDESS clips", "The reverse direction"],
    ], [3.7, 5.2, 4.2, 3.7]),
    Spacer(1, 3),
    Paragraph("<b>Metrics.</b> <i>Accuracy</i>; <i>macro-F1</i> (every emotion counts equally, because 20% of "
              "RAVDESS is neutral); <i>weighted F1</i> (only to compare with the paper); a <i>95% bootstrap "
              "confidence interval</i> of accuracy (2,000 resamples of the test clips); <i>McNemar's exact test</i> "
              "to compare two models on the same clips; and a <i>two-proportion z-test</i> plus a Mann–Whitney test "
              "over actors for the gender check. Chance is the majority class: 20.0% on RAVDESS, 14.3% on TESS.",
              body),
]

# 2. Results
rows = [["Model"] + ["P1 Random split", "P2 Augment first", "P3 Unseen speakers", "P4 RAV → TESS",
                     "P5 TESS → RAV"]]
for m in ["Majority", "SVM", "RandomForest", "CNN", "CNN_BiLSTM", "Ensemble"]:
    rows.append([NAMES[m]] + [f"{acc_ci(e, m)}<br/>F1 {100 * row(e, m)['macro_F1']:.1f}" for e in EXPS])
story += [
    KeepTogether([
        Paragraph("2. Results and Comparison of Models", h2),
        table(rows, [2.9, 2.8, 2.8, 2.8, 2.75, 2.75], bold_last=True),
        Paragraph("Table 1. Test accuracy % [95% confidence interval] and macro-F1 %. P3 is pooled over the four "
                  "speaker folds (fold std. of the ensemble: 1.9 points).", cap),
    ]),
    figure("fig1_accuracy_ci", 15.5, "Figure 1. Accuracy with 95% confidence intervals; dotted line = chance."),
    bullets([
        "<b>Same models, very different numbers.</b> The ensemble scores " + pct("leaky", "Ensemble") + " when "
        "augmented copies of test clips are in training (P2, close to the paper's 97.57%), "
        + pct("paper", "Ensemble") + " with a clean random split, " + pct("speaker", "Ensemble")
        + " for unseen speakers, and near chance on another dataset. The protocol matters more than the model.",
        "<b>Deep vs classical.</b> On the random split the ensemble beats the SVM by 4.9 points, but the test set "
        "has only 144 clips, the confidence intervals overlap and McNemar's test finds no significant difference ("
        + mcn("Random split (paper)", "Ensemble vs SVM") + "). With 1,440 unseen-speaker clips the advantage is "
        "clear: +9.4 points (" + mcn("Unseen speakers", "Ensemble vs SVM") + "). So the deep models do learn "
        "cues that transfer to new voices better than simple statistics.",
        "<b>Ensemble vs its members.</b> Averaging helps a little: +1.8 points over the 1D-CNN for unseen speakers ("
        + mcn("Unseen speakers", "Ensemble vs CNN") + "); the CNN-BiLSTM is not better than the 1D-CNN (p = 0.95).",
        "<b>Unseen dataset.</b> Every model is near chance on TESS, and the Random Forest is best ("
        + pct("ravdess_to_tess", "RandomForest") + "): its mean/std summary does not depend on <i>when</i> a "
        "sound happens in the clip, unlike the deep models' 3,672-value input.",
        "<b>Cost.</b> 1D-CNN 616,615 and CNN-BiLSTM 688,807 parameters; training took 10.6 and 13.1 min on a "
        "4-core CPU (21 and 23 epochs) and prediction 2.7 and 4.1 ms per clip, against 0.3 ms for the SVM, which "
        "trains in seconds.",
    ]),
]

# 3. Error analysis
story += [
    Paragraph("3. Error Analysis: Where the Model Does Well and Where It Fails", h2),
    figure("fig2_confusion", 16.5, "Figure 2. Ensemble confusion matrices (% of each true emotion). Left: random "
           "split. Middle: unseen speakers. Right: unseen dataset (TESS)."),
    bullets([
        "<b>Per emotion.</b> Under the random split neutral (93.1% recall), disgust (89.5%) and angry (84.2%) are "
        "easiest; happy and surprised are hardest (73.7%). For unseen speakers the order is angry (73.4%), "
        "surprised (67.7%) and neutral (64.2%) best, and <b>sad is worst (34.4%)</b>.",
        "<b>Typical mistakes are between similar-sounding emotions.</b> For unseen speakers the most common errors "
        "are sad → neutral (53 clips, 27.6% of sad clips), neutral → sad (39), neutral → disgust (28), "
        "happy → fearful (26) and angry → disgust (24). Low-energy emotions (sad, neutral) differ by small "
        "acoustic changes that are smaller than the differences between two people's voices.",
        "<b>Intensity.</b> Strongly expressed emotions are much easier: 66.1% vs 47.9% for normal intensity "
        "(unseen speakers, neutral excluded, 576 clips each, p &lt; 0.001).",
        "<b>Speakers.</b> Accuracy ranges from 40.0% (actors 9, 17 and 23) to 81.7% (actor 8) "
        "(Figure 3), so the result depends heavily on whose voice it is.",
        "<b>New dataset.</b> On TESS the ensemble predicts <i>sad</i> for 53.1% of all clips and almost never "
        "neutral, happy or surprised (recall 0.2%, 0.8%, 2.0%). TESS clips are short, so after the paper's 0.6 s "
        "offset 41.5% of their feature frames are zero padding, which looks like quiet, sad speech. Loading TESS "
        "from 0 s (18.0% padding) raises accuracy from 19.7% to 30.5% and cuts sad predictions to 11.1%, but the "
        "model stays far below its RAVDESS level. The older speaker is recognised worse (15.5% vs 23.9%).",
        "<b>Why P2 looks perfect.</b> When augmenting before splitting, 99.1% of test samples have a noisy or "
        "pitch-shifted copy of the same recording in training; accuracy is 97.8% on original clips and 95.6% on "
        "noise+pitch copies, so the model recognises recordings it has already seen.",
    ]),
]

# 4. Bias
brow = [["Model", "Male actors", "Female actors", "Gap (F − M)", "p-value"]]
for _, r in BIAS.iterrows():
    p = "&lt; 0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
    brow.append([NAMES[r["model"]], f"{r['male acc']:.1f}% (F1 {r['male macro-F1']:.1f})",
                 f"{r['female acc']:.1f}% (F1 {r['female macro-F1']:.1f})", f"+{r['gap (F - M)']:.1f}", p])
story += [
    KeepTogether([
        Paragraph("4. Bias Check: Male vs Female Speakers", h2),
        Paragraph("We use the unseen-speaker protocol, so every clip is judged by a model that never heard that "
                  "actor. RAVDESS is balanced (12 male and 12 female actors, 720 clips each), so a gap cannot be "
                  "caused by unequal training data.", body),
        table(brow, [3.2, 3.9, 3.9, 2.6, 2.2], bold_last=True),
        Paragraph("Table 2. Accuracy and macro-F1 by gender (720 clips per group, two-proportion z-test).", cap),
    ]),
    figure("fig3_actors", 15.5, "Figure 3. Ensemble accuracy for each actor (blue = male, orange = female)."),
    bullets([
        "<b>Every model is worse for male voices</b>, by 7.6 (SVM) to 18.6 points (CNN-BiLSTM); the ensemble "
        "gap is 15.6 points (50.7% vs 66.2%). The deep models, although more accurate overall, have the "
        "<i>largest</i> gaps.",
        "<b>It is not caused by a few actors.</b> Clips of one actor are not independent, so we also compare the 12 "
        "per-actor accuracies of each gender: Mann–Whitney U = 129 (out of 144), p = 0.001. Ten of the 12 lowest "
        "actors are male (Figure 3).",
        "<b>Per emotion</b> the gap is largest for surprised (+31.2 points), neutral (+20.1) and disgust (+19.8), "
        "and reversed only for happy (−2.1).",
        "With the random split the gap is smaller and not significant (80.0% vs 84.4%, p = 0.50), because the "
        "model has heard every actor in training. The bias appears exactly in the realistic case of a new user.",
    ]),
]

# 5. Insights + 6. challenges
story += [
    KeepTogether([Paragraph("5. Key Insights and Observations", h2), bullets([
        "The paper's 97.57% is reproduced only when augmented copies of test clips leak into training (96.5%). With "
        "a clean split the same models reach 81.9%, and for new speakers 58.5%: the headline number measures "
        "familiar recordings and voices, not emotion recognition in general.",
        "The deep ensemble is the best model whenever the test data come from the same corpus, and its advantage "
        "over the SVM is only provable with a large test set (unseen speakers). A 144-clip test set cannot "
        "reliably separate models that are only a few points apart.",
        "Nothing transfers across datasets: all models are near chance on TESS, partly because of zero padding "
        "from the fixed 0.6 s offset and partly because speakers, words and recording conditions differ.",
        "The model is significantly less accurate for male speakers. Before deployment, accuracy must be checked "
        "per group, and speaker normalisation should be tested as a fix (planned for Deliverable 5).",
    ])]),
    KeepTogether([Paragraph("6. Challenges Faced", h2), bullets([
        "<b>Small test set.</b> 144 test clips give ±6 point confidence intervals, so a single run can make "
        "models look better or worse than they are (our Phase 3 re-run gave the ensemble 77.1%, inside this "
        "run's interval of 75.7–88.2%). We added confidence intervals, McNemar tests and the 1,440-clip "
        "speaker-independent protocol to make comparisons reliable.",
        "<b>Dependent test clips.</b> Each actor contributes 60 clips, so the clip-level gender test overstates "
        "certainty; we added an actor-level test.",
        "<b>Explaining the TESS collapse.</b> Low accuracy alone does not say why; looking at the prediction "
        "distribution and the share of padded frames led to the controlled offset check.",
        "<b>Compute.</b> The 23 deep-model training runs behind these predictions take hours on a CPU, so the "
        "evaluation notebook works from saved predictions and can be re-run in seconds.",
    ])]),
    Paragraph("<b>To reproduce:</b> open " + code("phase4/F26-11.ipynb") + " and run all cells (needs pandas, "
              "scikit-learn, scipy, matplotlib; no GPU). The predictions themselves are regenerated by "
              + code("python scripts/run_experiments.py paper leaky speaker cross") + ". "
              "<b>Reference:</b> J. H. Chowdhury, S. Ramanna, K. Kotecha, <i>Sci. Rep.</i> 15, 11824 (2025).", body),
]

SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm,
                  bottomMargin=1.5 * cm, title="F26-11 Deliverable 4 – Evaluation and Analysis of Results",
                  author="F26-11 (22L-7786, 23L-3002, 23L-3102)").build(story)
print("wrote", OUT)
