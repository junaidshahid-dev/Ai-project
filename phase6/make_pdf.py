"""Build the final technical report phase6/F26-11.pdf (Deliverable 6). Numbers come from results/summary.json,
phase4/ and phase5/ (all produced by the executed notebooks and scripts in this repository)."""
import json
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
ROOT = HERE.parent
OUT = HERE / "F26-11.pdf"
S = json.load(open(ROOT / "results" / "summary.json"))
M = pd.read_csv(ROOT / "phase4" / "metrics.csv")
BIAS = pd.read_csv(ROOT / "phase4" / "gender_bias.csv")
NORM = pd.read_csv(ROOT / "phase5" / "speaker_normalisation.csv", index_col=0)
SHIFT = pd.read_csv(ROOT / "phase5" / "time_shift.csv", index_col=0)
REP = pd.read_csv(ROOT / "phase5" / "reproducibility.csv")
ABL = pd.DataFrame(S["ablation"])

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13, spaceAfter=3,
                      alignment=4)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=15, leading=19, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, alignment=1, fontSize=10, textColor=colors.HexColor("#333333"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12.5, spaceBefore=7, spaceAfter=3)
h3 = ParagraphStyle("h3", parent=body, fontName="Times-Bold", fontSize=10.5, spaceBefore=3, spaceAfter=1)
cell = ParagraphStyle("c", parent=body, fontSize=9, leading=10.8, spaceAfter=0, alignment=0)
cap = ParagraphStyle("cap", parent=body, fontSize=9, leading=11, alignment=1, textColor=colors.HexColor("#333333"))
abstract = ParagraphStyle("a", parent=body, fontSize=10, leading=12.5, leftIndent=12, rightIndent=12)
mono = ParagraphStyle("m", parent=cell, fontName="Courier", fontSize=8.6, leading=10.5)
code = "<font name='Courier' size='9'>{}</font>".format
NAMES = {"Majority": "Majority (chance)", "SVM": "SVM", "RandomForest": "Random Forest", "CNN": "1D-CNN",
         "CNN_BiLSTM": "CNN-BiLSTM", "Ensemble": "Ensemble"}
EXPS = ["paper", "leaky", "speaker", "ravdess_to_tess", "tess_to_ravdess"]


def row(exp, model):
    return M[(M["experiment"] == exp) & (M["model"] == model)].iloc[0]


def pct(x):
    return f"{100 * x:.1f}"


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=12) for t in items],
                        bulletType="bullet", start="•", leftIndent=14)


def numbered(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=14) for t in items], bulletType="1",
                        bulletFormat="%s.", bulletFontName="Times-Roman", bulletFontSize=10.5, leftIndent=16)


def table(rows, widths, bold_last=False, style_cell=cell):
    t = Table([[Paragraph(str(c), style_cell if i else cell) for c in r] for i, r in enumerate(rows)],
              colWidths=[w * cm for w in widths])
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8"))]
    if bold_last:
        style.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f3f7ff")))
    t.setStyle(TableStyle(style))
    return t


def image(path, width_cm):
    w, h = Image.open(path).size
    return RLImage(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def figure(path, width_cm, caption):
    return KeepTogether([image(ROOT / path, width_cm), Paragraph(caption, cap), Spacer(1, 3)])


def link(u):
    return f'<link href="{u}" color="blue">{u}</link>'


plain, norm = NORM.loc["without"], NORM.loc["with speaker normalisation"]
REPO = "https://github.com/junaidshahid-dev/Ai-project"
APP = "https://f26-11-speech-emotion.streamlit.app"

# ---------------------------------------------------------------- title + abstract
story = [
    Paragraph("National University of Computer &amp; Emerging Sciences, Lahore", sub),
    Paragraph("AI2002 – Artificial Intelligence (BS-SE-5A) — Project Deliverable 6: Final Technical Report", sub),
    Spacer(1, 5),
    Paragraph("How Far Does a Near-Perfect Speech Emotion Recogniser Generalise? Replication, Stress-Testing and "
              "Bias Analysis of a Lightweight Hand-Crafted-Feature Ensemble", h1),
    Spacer(1, 3),
]
info = [
    [Paragraph("<b>Group ID</b>", cell), Paragraph("F26-11", cell),
     Paragraph("<b>Track</b>", cell), Paragraph("B – Research &amp; Development", cell)],
    [Paragraph("<b>Members</b>", cell), Paragraph("22L-7786 (Lead), 23L-3002, 23L-3102", cell),
     Paragraph("<b>Instructor</b>", cell), Paragraph("Hajra Waheed", cell)],
    [Paragraph("<b>Repository</b>", cell), Paragraph(link(REPO) + " — code, notebooks, trained models, demo "
                                                      "app, README with setup and usage", cell), "", ""],
    [Paragraph("<b>Web app</b>", cell), Paragraph(link(APP) + " — upload a .wav file or pick an example clip "
                                                   "to see the predicted emotion", cell), "", ""],
]
t = Table(info, colWidths=[2.3 * cm, 6.3 * cm, 2.2 * cm, 6.0 * cm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("SPAN", (1, 2), (3, 2)), ("SPAN", (1, 3), (3, 3)),
                       ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
                       ("BACKGROUND", (2, 0), (2, 1), colors.HexColor("#eeeeee"))]))
story += [t, Spacer(1, 6), Paragraph("Abstract", h2), Paragraph(
    "Chowdhury et al. (<i>Scientific Reports</i>, 2025) report 97.57% accuracy on RAVDESS for a lightweight "
    "averaging ensemble of a 1D-CNN and a CNN-BiLSTM trained on four hand-crafted acoustic features (ZCR, RMS "
    "energy, MFCC, Chroma). We re-implemented the pipeline from the paper's specification and tested it under five "
    "evaluation protocols, against classical baselines, with a gender-bias check, a feature ablation and follow-up "
    "experiments that explain the results. With augmentation applied only to training clips, the ensemble reaches "
    "<b>81.9%</b>; applying augmentation <i>before</i> the train–test split, so that 99.1% of test samples have a "
    "near-copy in training, reproduces the paper's level (<b>96.5%</b>). For speakers never heard in training "
    "accuracy falls to <b>58.5% ± 1.9</b>, and on another corpus it is near chance (19.7% RAVDESS → TESS). All "
    "models are significantly less accurate for male than for female speakers (ensemble 50.7% vs 66.2%, p &lt; "
    "0.001). We show that the features identify the speaker (95.0%) better than the emotion (74.4%), that "
    "per-speaker normalisation raises unseen-speaker accuracy of an SVM from 49.3% to 59.2% without closing the "
    "gender gap, and that the paper's flattened input is sensitive to time shifts. The architecture is a sound "
    "lightweight model, but its headline accuracy is a property of the evaluation protocol.", abstract)]

story += [Paragraph("Summary of the project", h3), bullets([
    "<b>Goal:</b> check whether a published, near-perfect speech emotion recogniser still works on new clips, new "
    "speakers, a new dataset, and equally well for men and women (Track B: replication and analysis).",
    "<b>What we built:</b> the full pipeline (data indexing, feature extraction, augmentation, 5 models), 6 "
    "experiment protocols with 23 deep-model training runs, an evaluation and bias analysis, explanatory "
    "experiments, trained model files and a Streamlit demo app that predicts the emotion of an uploaded .wav file.",
    "<b>Main answer:</b> no. The reported accuracy depends on test clips leaking into training; for new speakers "
    "and new datasets the model is far weaker, and it is biased against male voices.",
])]

# ---------------------------------------------------------------- 1 introduction
story += [
    Paragraph("1. Introduction and Research Question", h2),
    Paragraph("Speech emotion recognition (SER) classifies a speaker's emotional state from the voice alone, with "
              "uses in call-centre analytics, health screening and voice assistants. Lightweight models are "
              "attractive because they run on cheap hardware. The base paper's numbers (97.57% on RAVDESS, 100% on "
              "TESS) were obtained by training and testing on random splits of the same dataset, it does not say "
              "whether augmentation was applied before or after splitting, and the authors list cross-dataset "
              "validation as future work. <b>Research question:</b> <i>does the near-perfect accuracy of this "
              "lightweight model reflect emotion recognition that transfers to new clips, new speakers and new "
              "recording conditions, and is it equally accurate for male and female speakers?</i> We test six "
              "hypotheses:", body),
    table([
        ["", "Hypothesis", "Result"],
        ["H1", "Under the paper's protocol our implementation approaches the reported accuracy.", "Not supported (81.9% vs 97.57%)"],
        ["H2", "Augmenting before splitting inflates accuracy.", "Supported (96.5%)"],
        ["H3", "Accuracy drops substantially for unseen speakers.", "Supported (58.5%)"],
        ["H4", "Accuracy collapses on an unseen dataset.", "Supported (19.7% / 14.1%)"],
        ["H5", "Accuracy differs between male and female speakers.", "Supported (−15.6 points for men)"],
        ["H6", "MFCC carries most information; Chroma adds a positive contribution.", "Partly (MFCC yes, Chroma not confirmed)"],
    ], [1.0, 10.2, 5.6]),
]

# ---------------------------------------------------------------- 2 data
story += [
    Paragraph("2. Data", h2),
    bullets([
        "<b>RAVDESS</b> (Kaggle): 1,440 speech clips from 24 professional actors (12 male, 12 female) speaking two "
        "fixed sentences in 8 emotions at normal and strong intensity, 48 kHz.",
        "<b>TESS</b> (Kaggle): 2,800 clips of “Say the word ___” with 200 words, spoken by two female actors (aged "
        "26 and 64) in 7 emotions. Used as the unseen dataset.",
        "<b>Common labels.</b> As in the paper (288 neutral RAVDESS clips in its Table 1), RAVDESS <i>calm</i> is "
        "merged into <i>neutral</i> and TESS <i>pleasant surprise</i> becomes <i>surprised</i>: 7 classes (neutral, "
        "happy, sad, angry, fearful, disgust, surprised). RAVDESS is balanced except neutral (288 vs 192 clips); "
        "TESS has 400 per class. Duplicated files in the Kaggle copies are removed by file name.",
    ]),
]

# ---------------------------------------------------------------- 3 methodology
story += [
    Paragraph("3. Methodology and Architecture", h2),
    figure("phase3/figures/fig0_pipeline.png", 16.5, "Figure 1. Pipeline from a .wav file to a predicted emotion. "
           "Top: the paper's deep models; bottom: our classical baselines."),
    Paragraph("<b>Features and augmentation (paper Section 4).</b> Audio is resampled to 22,050 Hz and 2.5 s are "
              "loaded from an offset of 0.6 s. With a 2,048-sample window and 512-sample hop this gives 108 frames; "
              "per frame we compute ZCR (1), RMS energy (1), 20 MFCCs and 12 Chroma values with librosa. Short "
              "clips are zero-filled. The 108 × 34 matrix is flattened to the paper's 3,672-value vector and "
              "z-scored with training-set statistics only. Each training clip is used in 4 versions: original, "
              "Gaussian noise, pitch shift (+0.7 semitones) and noise + pitch. Validation and test clips are never "
              "augmented (except in the deliberately leaky protocol).", body),
    figure("phase6/figures/fig_architecture.png", 17.0, "Figure 2. Architecture of the paper's 1D-CNN and "
           "CNN-BiLSTM (Table 2 of the paper) with the output size of every layer."),
    Paragraph("<b>Models.</b> (i) <b>1D-CNN</b>: five blocks of Conv1D → BatchNorm → MaxPool (128, 128, 64, 64, 32 "
              "filters), dropout 0.2 after blocks 2, 4 and 5, then Dense(128) and a 7-way softmax. (ii) "
              "<b>CNN-BiLSTM</b>: the same with a bidirectional LSTM (64 units per direction) before the last block. "
              "(iii) <b>Ensemble</b>: average of the two models' probabilities, the paper's final model. "
              "(iv) <b>Baselines</b>: majority class, an RBF-SVM (C = 10) and a 300-tree Random Forest on a 68-value "
              "summary (mean and std of each feature over time).", body),
    table([
        ["Training setting (paper Table 3)", "Value"],
        ["Optimiser, loss", "Adam (learning rate 0.001), sparse categorical cross-entropy"],
        ["Batch size, epochs", "64, up to 100"],
        ["Early stopping", "on validation accuracy, patience 5, best weights restored"],
        ["Learning-rate schedule", "halved after 3 epochs without improvement (minimum 1e-5)"],
        ["Seeds", "42 for every split, model and library"],
        ["Parameters", "1D-CNN 616,615; CNN-BiLSTM 688,807 (paper: 672,423, see Section 10)"],
    ], [5.5, 11.3]),
    Spacer(1, 4),
    Paragraph("<b>Evaluation protocols.</b>", body),
    table([
        ["Protocol", "Training data", "Test data", "Tests"],
        ["P1 Random split (paper)", "80% of RAVDESS clips + augmented copies; 10% validation", "144 clips", "H1"],
        ["P2 Augment before split", "80% of all 5,760 augmented samples", "576 samples", "H2"],
        ["P3 Unseen speakers", "4 folds; 18 actors train, 6 held out (3 M + 3 F)", "all 1,440 clips once", "H3, H5"],
        ["P4 / P5 Cross-dataset", "all RAVDESS (or all TESS)", "all TESS (or RAVDESS)", "H4"],
        ["Ablation", "P1 split, one feature group removed / MFCC only / no augmentation", "144 clips", "H6"],
    ], [4.0, 6.8, 3.6, 2.4]),
    Paragraph("<b>Metrics:</b> accuracy, macro-F1 (each emotion counts equally), weighted F1 (to compare with the "
              "paper), 95% bootstrap confidence intervals, McNemar's test between models, and a two-proportion "
              "z-test plus an actor-level Mann–Whitney test for the gender check.", body),
]

# ---------------------------------------------------------------- 4 implementation
story += [
    Paragraph("4. Implementation Steps and Workflow", h2),
    Paragraph("Python 3.13 with librosa 0.11 (audio and features), NumPy and pandas (arrays and tables), "
              "scikit-learn (baselines, splits, metrics), TensorFlow 2.21 / Keras 3 (deep models), SciPy (tests), "
              "Matplotlib (figures) and Streamlit (demo). All experiments run on CPU.", body),
    table([
        ["Step", "Code", "What happens"],
        ["1. Index data", code("src/ser/data.py"), "find all .wav files, remove duplicates, map labels, read actor and gender"],
        ["2. Features", code("scripts/extract_features.py"), "4 versions × 108 frames × 34 features per clip, saved once to features/*.npz"],
        ["3. Models", code("src/ser/models.py"), "baselines and the paper's 1D-CNN / CNN-BiLSTM / ensemble"],
        ["4. Experiments", code("scripts/run_experiments.py"), "trains every model under every protocol, saves per-clip predictions in results/"],
        ["5. Evaluation", code("phase4/F26-11.ipynb"), "metrics, confidence intervals, model comparison, errors, gender bias"],
        ["6. Analysis", code("phase5/F26-11.ipynb"), "reproducibility check and follow-up experiments"],
        ["7. Use the model", code("app/app.py"), "upload a .wav file → features → normalise → both models → average → emotion"],
    ], [2.6, 5.9, 8.3]),
    Spacer(1, 3),
    Paragraph("The deliverable notebooks follow the same steps in a simpler, self-contained form: Phase 2 "
              "(data and features), Phase 3 (training the four models and predicting a raw file), Phase 4 "
              "(evaluation) and Phase 5 (analysis). The final models used by the app are trained on all of RAVDESS "
              "and saved as " + code("models/CNN.keras") + ", " + code("models/CNN_BiLSTM.keras") + " and "
              + code("models/normalizer.npz") + " (training mean and std).", body),
]

# ---------------------------------------------------------------- 5 results
rows = [["Model", "P1 Random split", "P2 Augment first", "P3 Unseen speakers", "P4 RAV → TESS", "P5 TESS → RAV"]]
for m in ["Majority", "SVM", "RandomForest", "CNN", "CNN_BiLSTM", "Ensemble"]:
    rows.append([NAMES[m]] + [f"{pct(row(e, m)['accuracy'])} [{100 * row(e, m)['ci_low']:.0f}–"
                              f"{100 * row(e, m)['ci_high']:.0f}]<br/>F1 {pct(row(e, m)['macro_F1'])}" for e in EXPS])
story += [
    KeepTogether([
        Paragraph("5. Results and Evaluation Summary", h2),
        table(rows, [2.9, 2.8, 2.8, 2.8, 2.75, 2.75], bold_last=True),
        Paragraph("Table 1. Test accuracy % [95% confidence interval] and macro-F1 %. Chance (majority class) is "
                  "20.0% on RAVDESS and 14.3% on TESS.", cap),
    ]),
    figure("phase4/figures/fig1_accuracy_ci.png", 15.5, "Figure 3. Accuracy of every model under every protocol "
           "with 95% confidence intervals."),
    bullets([
        "<b>The protocol matters more than the model:</b> the same ensemble ranges from 96.5% (P2) to 81.9% (P1), "
        "58.5% (P3) and near chance (P4, P5).",
        "<b>Deep models help within a corpus:</b> for unseen speakers the ensemble beats the SVM by 9.4 points "
        "(McNemar p &lt; 0.001). On the 144-clip random split the 4.9-point lead is not significant (p = 0.30).",
        "<b>Ensembling helps a little:</b> +1.8 points over the 1D-CNN for unseen speakers (p = 0.04); the "
        "CNN-BiLSTM is not better than the 1D-CNN.",
        "<b>Efficiency:</b> on a 4-core CPU the 1D-CNN trained in 10.6 min (21 epochs) and the CNN-BiLSTM in 13.1 "
        "min (23 epochs); prediction takes 2.7 and 4.1 ms per clip, the SVM 0.3 ms. On our 2-core laptop one "
        "epoch takes about 3.7 min.",
    ]),
]

# ---------------------------------------------------------------- 6 replication vs paper
rep = {r["model"]: r for r in S["replication"]}
story += [
    KeepTogether([
        Paragraph("6. Replication Results vs Reported Paper Results", h2),
        table([
            ["Model (RAVDESS)", "Reported in paper: acc / weighted F1", "Our replication, clean split (P1)",
             "Our replication, augment before split (P2)"],
        ] + [[NAMES[m], f"{rep[m]['reported_acc']:.2f} / {rep[m]['reported_f1w']:.2f}",
              f"{rep[m]['ours_paper_acc']:.2f} / {rep[m]['ours_paper_f1w']:.2f}",
              f"{rep[m]['ours_leaky_acc']:.2f} / {rep[m]['ours_leaky_f1w']:.2f}"] for m in ["CNN", "CNN_BiLSTM", "Ensemble"]]
          + [["Ensemble on TESS", "100.00 / 100.00", "not tested within TESS", "–"]],
              [3.4, 4.3, 4.3, 4.8], bold_last=False),
        Paragraph("Table 2. Accuracy and weighted F1 (%) reported by Chowdhury et al. (Table 4) and obtained by us.",
                  cap),
    ]),
    bullets([
        "With a leak-free split our implementation is <b>15.6 points below</b> the reported ensemble accuracy. "
        "The order of the models matches the paper (ensemble ≥ CNN-BiLSTM ≥ 1D-CNN).",
        "Augmenting before splitting reproduces the paper's numbers within about one point (96.5% vs 97.57%; "
        "1D-CNN 95.7% vs 96.18%). In that setting 99.1% of test samples have a noisy or pitch-shifted copy of the "
        "same recording in training, and accuracy is 97.8% on original clips and 95.6% on noise + pitch copies.",
        "The paper does not state its split order, so we cannot prove that this is what happened, but it is the "
        "only one of our settings that reproduces its results. The inflation is much larger for the deep models "
        "(+14.6 points for the ensemble) than for the SVM (+8.5), consistent with memorising near-duplicates.",
        "Our CNN-BiLSTM has 688,807 parameters instead of 672,423: the paper's Table 2 lists 6,176 parameters for "
        "the last convolution, which implies 64 input channels although the Bi-LSTM before it outputs 128. We "
        "followed the layer description.",
    ]),
]

# ---------------------------------------------------------------- 7 error analysis
story += [
    KeepTogether([Paragraph("7. Error Analysis", h2),
                  figure("phase4/figures/fig2_confusion.png", 16.5, "Figure 4. Ensemble confusion matrices (% of each "
                         "true emotion): random split, unseen speakers, unseen dataset.")]),
    bullets([
        "<b>Easy and hard emotions.</b> For unseen speakers angry (73.4% recall), surprised (67.7%) and neutral "
        "(64.2%) are best; <b>sad is worst (34.4%)</b>. The most frequent errors are sad → neutral (27.6% of sad "
        "clips), neutral → sad, neutral → disgust, happy → fearful and angry → disgust: emotions with similar "
        "energy and pitch are confused.",
        "<b>Intensity:</b> strongly expressed emotions are recognised much better (66.1% vs 47.9%, p &lt; 0.001).",
        "<b>Speakers:</b> per-actor accuracy ranges from 40.0% to 81.7% (Figure 5).",
        "<b>New dataset:</b> on TESS the ensemble predicts <i>sad</i> for 53.1% of clips and almost never neutral, "
        "happy or surprised. TESS clips are short, so after the 0.6 s offset 41.5% of their frames are zero padding "
        "(RAVDESS: almost none), which looks like quiet, sad speech. Loading TESS from 0 s instead raises the "
        "same ensemble from 19.7% to 30.5% and cuts sad predictions to 11.1%; the rest of the gap comes from "
        "different speakers, words and recording conditions. The older TESS speaker is recognised worse "
        "(15.5% vs 23.9%).",
    ]),
]

# ---------------------------------------------------------------- 8 bias
brow = [["Model", "Male actors", "Female actors", "Gap (F − M)", "p-value"]]
for _, r in BIAS.iterrows():
    p = "&lt; 0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
    brow.append([NAMES[r["model"]], f"{r['male acc']:.1f}%", f"{r['female acc']:.1f}%",
                 f"+{r['gap (F - M)']:.1f}", p])
story += [
    KeepTogether([
        Paragraph("8. Bias and Fairness Check (Gender)", h2),
        Paragraph("Unseen-speaker protocol, so every clip is judged by a model that never heard that actor; 720 "
                  "clips per gender, so the data are balanced.", body),
        table(brow, [3.6, 3.3, 3.3, 3.3, 3.3], bold_last=True),
        Paragraph("Table 3. Accuracy by gender (two-proportion z-test).", cap),
    ]),
    figure("phase4/figures/fig3_actors.png", 15.0, "Figure 5. Ensemble accuracy per actor (blue = male, orange = "
           "female)."),
    Paragraph("Every model is worse for male voices, and the deep models have the largest gaps (up to 18.6 "
              "points). The result holds at actor level (12 vs 12 per-actor accuracies, Mann–Whitney U = 129, "
              "p = 0.001; 10 of the 12 lowest actors are male). The gap is largest for surprised (+31.2), neutral "
              "(+20.1) and disgust (+19.8) and reversed only for happy (−2.1). With a random split, where every "
              "actor is in training, the gap is small and not significant (80.0% vs 84.4%, p = 0.50): the bias "
              "appears exactly in the realistic case of a new user.", body),
]

# ---------------------------------------------------------------- 9 ablation
abl = ABL.set_index(["variant", "model"])["acc"]
story += [
    Paragraph("9. Feature and Augmentation Ablation", h2),
    figure("results/figures/fig_ablation.png", 10.5, "Figure 6. Change in 1D-CNN accuracy when parts of the input "
           "are removed (random split, 144 test clips; one standard error ≈ 3.4 points). *With the paper's early stopping, which halts before the model learns."),
    bullets([
        f"<b>MFCC is essential:</b> without it the 1D-CNN falls from {pct(abl['full', 'CNN'])}% to "
        f"{pct(abl['no_mfcc', 'CNN'])}% and the SVM from {pct(abl['full', 'SVM'])}% to {pct(abl['no_mfcc', 'SVM'])}%. "
        f"MFCC alone ({pct(abl['mfcc_only', 'CNN'])}%) is as good as all features.",
        "<b>ZCR and RMS add nothing measurable</b> (changes of +0.7 and 0.0 points).",
        f"<b>Chroma is inconsistent:</b> removing it changes the 1D-CNN by "
        f"{100 * (abl['no_chroma', 'CNN'] - abl['full', 'CNN']):+.1f} and the SVM by "
        f"{100 * (abl['no_chroma', 'SVM'] - abl['full', 'SVM']):+.1f} points, within noise; the paper's claim "
        "that Chroma contributes substantially is not confirmed.",
        f"<b>Augmentation and early stopping interact:</b> without augmentation the 1D-CNN stopped at epoch 9 "
        f"with validation accuracy still at chance ({pct(abl['no_augmentation', 'CNN'])}%), because an epoch has "
        f"4× fewer updates. With patience 20 it reaches {pct(abl['no_augmentation_patience20', 'CNN'])}%, so "
        "augmentation helps by only about 5 points.",
    ]),
]

# ---------------------------------------------------------------- 10 theory
story += [
    Paragraph("10. Theoretical Insights (Follow-up Experiments)", h2),
    Paragraph("To explain the results we ran four cheap experiments with the SVM on the same features "
              "(" + code("phase5/F26-11.ipynb") + ").", body),
    bullets([
        "<b>The features describe the speaker more than the emotion.</b> On average the speaker explains 27.9% of "
        "a feature's variance and the emotion 7.6% (eta squared; speaker dominates in 65 of 68 features). The same "
        "SVM identifies <b>who is speaking with 95.0%</b> accuracy (chance 4.2%) but the emotion with 74.4%. A "
        "model can therefore score well on familiar speakers by learning each voice's typical values, which fail "
        "for a new voice.",
        f"<b>Per-speaker normalisation helps new speakers and new datasets.</b> Z-scoring each speaker's features "
        f"with that speaker's own statistics (no labels needed) raises the SVM from {pct(plain['unseen speakers'])}% "
        f"to <b>{pct(norm['unseen speakers'])}%</b> for unseen speakers (as good as the deep ensemble) and from "
        f"{pct(plain['RAVDESS -> TESS'])}% to <b>{pct(norm['RAVDESS -> TESS'])}%</b> on TESS. It does <i>not</i> close "
        f"the gender gap ({100 * plain['gap (F - M)']:.1f} → {100 * norm['gap (F - M)']:.1f} points), so the bias has "
        "another cause than different voice baselines.",
        f"<b>The flattened input is tied to time positions.</b> Delaying the test speech by only 0.12 s costs an "
        f"SVM on the paper's 3,672-value input {100 * (SHIFT.iloc[0, 0] - SHIFT.iloc[1, 0]):.1f} points but an SVM "
        f"on the 68-value summary only {100 * (SHIFT.iloc[0, 1] - SHIFT.iloc[1, 1]):.1f}. This explains why the "
        "summary-based Random Forest transfers best to TESS and why padding hurts the deep models.",
        "<b>A 144-clip test set is noisy.</b> Ten random splits give the same SVM 72.9%–83.3% (std 4.1 points; "
        "theory: 3.5). Differences of a few points between models or runs are within this noise, which is why we "
        "rely on confidence intervals and the 1,440-clip speaker protocol.",
    ]),
    figure("phase5/figures/fig1_speaker_vs_emotion.png", 10.5, "Figure 7. Share of feature variance explained by "
           "the speaker and by the emotion."),
]

# ---------------------------------------------------------------- 11 demo + reproducibility
story += [
    Paragraph("11. Working Product and Reproducibility", h2),
    Paragraph("<b>Web app.</b> The app is hosted online at " + link(APP) + " (or run locally with "
              + code("streamlit run app/app.py") + "). The user uploads a .wav file or picks one of five example "
              "clips (three RAVDESS, two from the unseen TESS dataset); the app extracts the same 3,672 features, normalises them with the saved training "
              "statistics, runs both saved models, averages their probabilities and shows the predicted emotion "
              "with a bar chart of all 7 probabilities, plus a warning that accuracy is lower for other speakers and "
              "recording conditions. The features it computes from an uploaded file are identical to the cached "
              "training features (maximum difference 0.0). In Phase 3, the notebook's prediction function classified "
              "5 of 6 raw test files correctly.", body),
    Paragraph(f"<b>Reproducibility.</b> All seeds are fixed (42) and every per-clip prediction is saved. All 30 "
              f"reported accuracies are recomputed from these files, and re-training the SVM on another machine "
              f"reproduces {pct(REP.loc[0, 'same predictions'])}% of its random-split predictions and "
              f"{pct(REP.loc[2, 'same predictions'])}% of its unseen-speaker predictions. Deep models and the Random "
              "Forest vary by a few points with hardware and library versions (Phase 3 re-run: ensemble 77.1%, "
              "inside the 95% interval of 75.7–88.2%).", body),
]

# ---------------------------------------------------------------- 12 challenges
story += [
    Paragraph("12. Challenges Faced", h2),
    bullets([
        "<b>Environment setup.</b> TensorFlow 2.21 is about 1.5 GB; on our slow connection pip's dependency resolver "
        "appeared to hang until we pinned the exact version. Jupyter first started the system Python without "
        "TensorFlow, so we had to register a kernel for the correct environment. The demo app's temporary-file "
        "handling did not work on Windows and was changed to read the upload from memory.",
        "<b>Compute.</b> Convolving a 3,672-step input with 128 filters is slow without a GPU: about 3.7 min per epoch "
        "on our 2-core laptop, so a single training of both deep models takes about 2.8 hours. The 23 runs of "
        "the full study took roughly 4 hours on a 4-core machine; the ablation therefore uses one split, and the "
        "follow-up experiments use the fast SVM.",
        "<b>Training convergence.</b> Both deep models overfit: training accuracy reaches about 99% within 10 "
        "epochs while validation accuracy levels off at 70–80%. Dropout, batch normalisation, early stopping and "
        "learning-rate halving keep the best epoch. Without augmentation the paper's early-stopping patience "
        "stopped training before the model learned anything (31.2%); a longer patience fixed it.",
        "<b>Under-specified paper.</b> Split order, ensemble training and the label of RAVDESS <i>calm</i> are not "
        "stated, and the parameter table is inconsistent; we made each choice explicit and tested the split order.",
        "<b>Data issues.</b> Duplicate files in the Kaggle copies (removed by name), the unbalanced neutral class "
        "(hence macro-F1), and short TESS clips that become mostly zero padding.",
        "<b>Honest evaluation.</b> A 144-clip test set cannot separate models a few points apart, and clips of one "
        "actor are not independent; we added confidence intervals, McNemar and actor-level tests.",
    ]),
    Paragraph("<b>Limitations and future work.</b> Both corpora contain acted North-American English speech, so even "
              "our strictest results overestimate performance on natural speech; TESS has only two (female) "
              "speakers, so the gender analysis uses RAVDESS only. Next steps are per-speaker normalisation inside "
              "the deep models, features that do not depend on time position, multi-corpus training and a larger, "
              "more diverse corpus such as CREMA-D (91 speakers).", body),
]

# ---------------------------------------------------------------- 13 how to run
story += [
    KeepTogether([
        Paragraph("13. Instructions to Run the Code and Reproduce the Results", h2),
        Paragraph("Full instructions are in the repository README. In short (Python 3.13, CPU only):", body),
        table([["Command", "Purpose"]] + [[code(c), p] for c, p in [
               ("git clone " + REPO + ".git", "get the code"),
               ["pip install -r requirements.txt", "install the libraries (TensorFlow only needed for training / app)"],
               ["download RAVDESS and TESS from Kaggle into data/RAVDESS and data/TESS", "datasets (links in README)"],
               ["python scripts/extract_features.py", "build the feature cache (about 10 min)"],
               ["streamlit run app/app.py", "demo: upload a .wav file, see the predicted emotion"],
               ["python scripts/run_experiments.py paper leaky speaker cross ablation ablation_control",
                "all training runs and saved predictions (about 4 h on 4 cores)"],
               ["python scripts/analyze.py", "tables, figures and results/summary.json"],
               ["run phase4/F26-11.ipynb and phase5/F26-11.ipynb", "evaluation (under 1 min) and analysis (about 6 min)"]]],
              [9.6, 7.2]),
    ]),
]

# ---------------------------------------------------------------- 14 conclusion + refs
story += [
    Paragraph("14. Conclusion", h2),
    Paragraph("We re-implemented a recent lightweight speech emotion recogniser and asked how far its near-perfect "
              "accuracy generalises. Under a leak-free split it reaches 81.9%, not 97.57%; the reported level is "
              "reproduced only when augmented copies of test clips are in training. For new speakers accuracy drops "
              "to 58.5%, across datasets it is near chance, and it is significantly worse for male speakers. The "
              "features carry more information about the speaker than the emotion; per-speaker normalisation is a "
              "simple remedy for new speakers and datasets, while the gender gap needs further work. For "
              "practitioners the lesson is concrete: split before augmenting, report speaker-independent and "
              "cross-corpus results with confidence intervals, and check accuracy per demographic group before "
              "deploying an emotion recogniser.", body),
    Paragraph("References", h2),
    Paragraph("[1] J. H. Chowdhury, S. Ramanna, K. Kotecha, “Speech emotion recognition with light weight deep neural "
              "ensemble model using hand crafted features,” <i>Scientific Reports</i> 15, 11824 (2025), "
              "doi:10.1038/s41598-025-95734-z. [2] S. R. Livingstone, F. A. Russo, “The Ryerson Audio-Visual "
              "Database of Emotional Speech and Song (RAVDESS),” <i>PLoS ONE</i> 13(5), e0196391 (2018). "
              "[3] M. K. Pichora-Fuller, K. Dupuis, “Toronto Emotional Speech Set (TESS),” Borealis (2020). "
              "[4] B. McFee et al., “librosa: Audio and music signal analysis in Python,” <i>SciPy</i> (2015). "
              "[5] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” <i>JMLR</i> 12 (2011). "
              "[6] M. Abadi et al., “TensorFlow: Large-scale machine learning on heterogeneous systems” (2015). "
              "[7] S. Kapoor, A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based "
              "science,” <i>Patterns</i> 4(9) (2023).", body),
]

SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm,
                  bottomMargin=1.5 * cm, title="F26-11 Deliverable 6 – Final Technical Report",
                  author="F26-11 (22L-7786, 23L-3002, 23L-3102)").build(story)
print("wrote", OUT)
