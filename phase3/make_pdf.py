"""Build the Phase 3 report phase3/F26-11.pdf (numbers come from the executed notebook F26-11.ipynb)."""
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
RES = pd.read_csv(HERE / "results.csv", header=[0, 1], index_col=0)

# Values printed by the executed notebook (training log and prediction demo).
TRAIN = {"CNN": (616615, 18, 0.757, 71.9), "CNN-BiLSTM": (688807, 21, 0.799, 94.7)}
DEMO = ["neutral → neutral (79%)", "fearful → fearful (≈100%)", "fearful → sad (≈100%)",
        "angry → angry (82%)", "neutral → neutral (58%)", "neutral → neutral (42%)"]
OBSERVATIONS = [
    "<b>The model works on RAVDESS.</b> The ensemble classifies 111 of the 144 test clips correctly (77.1%, "
    "macro-F1 75.9%), almost four times the 20% chance level. In this run the deep models are level with the SVM "
    "(CNN 76.4%, CNN-BiLSTM and ensemble 77.1%, SVM 77.1%), and the Random Forest is weakest (67.4%). With 144 test clips one clip is worth 0.7 points, so gaps "
    "of a few points between models are within run-to-run variation (see <i>Reproducibility</i> in Section 6).",
    "<b>Training behaviour (Fig. 2).</b> Training accuracy reaches 99% by epoch 10 while validation accuracy "
    "levels off at about 70–80%, so both models overfit the 4,608 training samples. The learning rate was halved "
    "once for the CNN and twice for the CNN-BiLSTM; early stopping kept the best epoch (CNN 75.7% at epoch 13, "
    "CNN-BiLSTM 79.9% at epoch 16) and ended training 5 epochs later.",
    "<b>Per emotion (Fig. 3, left).</b> Neutral is always recognised (29 of 29); angry and disgust come next "
    "(16 of 19 each), while happy and surprised are hardest (12 of 19). Most errors go towards neutral: 4 sad "
    "and 3 fearful clips are predicted as neutral, so quieter emotions are easily mistaken for neutral speech.",
    "<b>Unseen dataset (TESS).</b> All models fall to 17–31% accuracy, close to the 14.3% chance level (ensemble "
    "21.0%). The models learned cues specific to RAVDESS (its speakers, recording setup and clip length: 41.5% of "
    "TESS feature frames are zero padding against 0.02% for RAVDESS) rather than general emotion cues. This "
    "generalisation gap is the central question of our project.",
]

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13.2, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=14.5, leading=18, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, alignment=1, fontSize=10, textColor=colors.HexColor("#333333"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12, spaceBefore=6, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=9.5, leading=11.5, spaceAfter=0)
cap = ParagraphStyle("cap", parent=body, fontSize=9, leading=11, alignment=1, textColor=colors.HexColor("#333333"))
code = "<font name='Courier' size='9.5'>{}</font>".format


def pct(x):
    return f"{100 * x:.1f}"


def acc(exp, model):
    return pct(RES.loc[model, (exp, "accuracy")])


def bullets(items, style=body):
    return ListFlowable([ListItem(Paragraph(t, style), leftIndent=12) for t in items],
                        bulletType="bullet", start="•", leftIndent=14)


def numbered(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=14) for t in items],
                        bulletType="1", bulletFormat="%s.", bulletFontName="Times-Roman", bulletFontSize=10.5,
                        leftIndent=16)


def table(rows, widths, header=True, bold_last=False):
    t = Table([[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=[w * cm for w in widths])
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")))
    if bold_last:
        style.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f3f7ff")))
    t.setStyle(TableStyle(style))
    return t


def image(name, width_cm):
    w, h = Image.open(FIG / f"{name}.png").size
    return RLImage(str(FIG / f"{name}.png"), width=width_cm * cm, height=width_cm * cm * h / w)


def figure(name, width_cm, caption):
    return KeepTogether([image(name, width_cm), Paragraph(caption, cap), Spacer(1, 4)])


def link(u):
    return f'<link href="{u}" color="blue">{u}</link>'


story = [
    Paragraph("National University of Computer &amp; Emerging Sciences, Lahore", sub),
    Paragraph("AI2002 – Artificial Intelligence (BS-SE-5A) — Project Phase 3: Model Implementation &amp; "
              "Initial Results", sub),
    Spacer(1, 5),
    Paragraph("Lightweight Speech Emotion Recognition Using Hand-Crafted Acoustic Features: "
              "Cross-Dataset Generalization and Gender-Bias Analysis", h1),
    Spacer(1, 3),
]
info = [
    [Paragraph("<b>Group ID</b>", cell), Paragraph("F26-11", cell),
     Paragraph("<b>Track</b>", cell), Paragraph("B – Research &amp; Development", cell)],
    [Paragraph("<b>Members</b>", cell), Paragraph("22L-7786 (Lead), 23L-3002, 23L-3102", cell),
     Paragraph("<b>Instructor</b>", cell), Paragraph("Hajra Waheed", cell)],
    [Paragraph("<b>Code</b>", cell), Paragraph("F26-11.ipynb (also F26-11.py) · "
               + link("https://github.com/junaidshahid-dev/Ai-project") + " (folder phase3/)", cell), "", ""],
]
t = Table(info, colWidths=[2.2 * cm, 6.4 * cm, 2.2 * cm, 6.0 * cm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("SPAN", (1, 2), (3, 2)),
                       ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
                       ("BACKGROUND", (2, 0), (2, 1), colors.HexColor("#eeeeee"))]))
story += [t, Spacer(1, 6)]

# 1. Model selection
story += [
    Paragraph("1. Model Selection and Justification", h2),
    Paragraph("The task is 7-class speech emotion recognition (neutral, happy, sad, angry, fearful, disgust, "
              "surprised) from the hand-crafted features prepared in Phase 2. Because this is a Track B project "
              "that re-examines a published method (Chowdhury et al., <i>Sci. Rep.</i> 15, 11824, 2025, which "
              "reports 97.57% on RAVDESS), our main models are the <b>paper's own lightweight deep models with its "
              "hyper-parameters</b>. Two classical baselines tell us whether the deep models add anything beyond "
              "simple feature statistics.", body),
    table([
        ["Model", "Input", "Why we chose it"],
        ["SVM, RBF kernel (C = 10)", "68 values: mean + std of the 34 features over time",
         "Standard strong baseline for small datasets; trains in about 20 seconds."],
        ["Random Forest (300 trees)", "same 68 values", "Non-linear, robust, needs almost no tuning."],
        [f"1D-CNN (paper; {TRAIN['CNN'][0]:,} parameters)", "3,672-value feature vector",
         "Convolutions learn local patterns along the feature vector; small enough to train on a CPU."],
        [f"CNN-BiLSTM (paper; {TRAIN['CNN-BiLSTM'][0]:,} parameters)", "3,672-value feature vector",
         "A bidirectional LSTM reads the convolved sequence in both directions to capture longer-range order."],
        ["Ensemble (paper's final model)", "probabilities of the two deep models",
         "Average of the two outputs; combining different models reduces their individual errors."],
    ], [4.6, 4.4, 7.8], bold_last=True),
    Spacer(1, 3),
    Paragraph("Larger pre-trained speech models (e.g. wav2vec 2.0) would probably score higher, but our research "
              "question is whether <i>this</i> lightweight model and its reported accuracy hold up on unseen "
              "speakers and an unseen dataset, so we keep the paper's architecture (its Table 2) and training "
              "settings (its Table 3) unchanged.", body),
]

# 2. Implementation details
story += [
    Paragraph("2. Implementation Details", h2),
    bullets([
        "<b>Python 3.13</b> in a Jupyter notebook; all training on a 2-core laptop CPU (Intel i7-6600U, no GPU). "
        "Every random seed is 42.",
        "<b>TensorFlow 2.21 / Keras 3</b>: the deep models (Conv1D, BatchNormalization, MaxPooling1D, Dropout, "
        "Bidirectional LSTM, Dense), the Adam optimiser and the EarlyStopping / ReduceLROnPlateau callbacks.",
        "<b>scikit-learn 1.6</b>: SVM, Random Forest, StandardScaler, stratified train_test_split and the metrics "
        "(accuracy, macro-F1, classification report, confusion matrix).",
        "<b>librosa 0.11</b>: audio loading and features for new .wav files; <b>NumPy / pandas</b> for arrays and "
        "tables; <b>Matplotlib</b> for figures.",
    ]),
    Paragraph("<b>Code structure</b> (one small function per step, all in " + code("F26-11.ipynb") + "):", body),
    table([
        ["Function", "What it does"],
        [code("load_features"), "loads the Phase 2 feature cache and the label table"],
        [code("random_split"), "makes the stratified 80/10/10 train / validation / test clip indices"],
        [code("get_frames") + ", " + code("to_vectors") + ", " + code("to_summary"),
         "select clips (with augmented copies for training) and build the 3,672-value or 68-value inputs"],
        [code("conv_block") + ", " + code("build_model"), "build the paper's 1D-CNN or CNN-BiLSTM"],
        [code("train_model") + ", " + code("train_all"), "fit a deep model; train all four models on one split"],
        [code("predict_all") + ", " + code("evaluate"), "class probabilities (and ensemble); accuracy and macro-F1"],
        [code("extract_features") + ", " + code("predict_emotion"), "raw .wav file → features → predicted emotion"],
    ], [7.0, 9.8]),
]

# 3. Pipeline
story += [
    Paragraph("3. Code Workflow / Pipeline", h2),
    figure("fig0_pipeline", 16.8, "Figure 1. Pipeline from a .wav file to a predicted emotion."),
    numbered([
        "<b>Input:</b> the Phase 2 cache " + code("features/ravdess.npz") + " (1,440 clips × 4 versions × 108 "
        "frames × 34 features) and " + code("features/tess.npz") + " (2,800 clips), with labels, actor and gender.",
        "<b>Split RAVDESS by clip</b> (80/10/10). Only training clips use their 3 augmented copies "
        "(noise, pitch shift, both), so no version of a test clip is seen in training. TESS is never trained on.",
        "<b>Model input:</b> deep models get the flattened 3,672-value vector, z-scored with the training "
        "mean / std; the baselines get the 68-value summary.",
        "<b>Train</b> the four models (" + code("train_all") + "), then <b>predict</b> class probabilities; "
        "the ensemble is the average of the CNN and CNN-BiLSTM probabilities.",
        "<b>Output:</b> the emotion with the highest probability and its confidence; on a test set, accuracy, "
        "macro-F1 and a confusion matrix. " + code("predict_emotion(path)") + " runs the whole chain on a raw file.",
    ]),
]

# 4. Training process
t1 = TRAIN
story += [
    Paragraph("4. Training Process", h2),
    Paragraph("<b>Data split</b> (stratified by emotion, seed 42; validation is only used for early stopping "
              "and the learning-rate schedule):", body),
    table([
        ["Experiment", "Train clips (samples)", "Validation", "Test", "Question"],
        ["1. RAVDESS 80/10/10 (paper)", "1,152 (4,608)", "144", "144 RAVDESS", "Does the model work?"],
        ["2. TESS (Exp. 1 models)", "—", "—", "2,800 TESS", "Works on a new corpus?"],
    ], [4.8, 3.3, 1.9, 2.6, 4.2]),
    Spacer(1, 4),
    KeepTogether([table([
        ["Setting (paper Table 3)", "Value", "Training run (Exp. 1)", "CNN", "CNN-BiLSTM"],
        ["Optimiser / loss", "Adam, lr 0.001 / cross-entropy", "Parameters",
         f"{t1['CNN'][0]:,}", f"{t1['CNN-BiLSTM'][0]:,}"],
        ["Batch / max. epochs", "64 / 100", "Epochs until early stop",
         f"{t1['CNN'][1]}", f"{t1['CNN-BiLSTM'][1]}"],
        ["Early stopping", "patience 5 on val. accuracy", "Best validation accuracy",
         f"{pct(t1['CNN'][2])}%", f"{pct(t1['CNN-BiLSTM'][2])}%"],
        ["Learning rate", "halved after 3 flat epochs", "Training time (CPU)",
         f"{t1['CNN'][3]:.0f} min", f"{t1['CNN-BiLSTM'][3]:.0f} min"],
        ["Regularisation", "dropout 0.2, batch norm.", "SVM / RF time", "20 s", "11 s"],
    ], [3.6, 4.6, 3.8, 2.0, 2.8]), Spacer(1, 4)]),
    figure("fig1_training_curves", 13.5, "Figure 2. Experiment 1 training curves (solid = training, dashed = "
           "validation)."),
]

# 5. Results
rows = [["Model", "Exp. 1: RAVDESS test (144 clips)", "Exp. 2: TESS, unseen dataset (2,800 clips)"]]
for m in RES.index:
    rows.append([m] + [f"{pct(RES.loc[m, (e, 'accuracy')])} / {pct(RES.loc[m, (e, 'macro F1')])}"
                       for e in ("RAVDESS test", "TESS (unseen dataset)")])
side = Table([[image("fig2_confusion_matrix", 7.4), image("fig3_results", 9.4)]],
             colWidths=[7.7 * cm, 9.6 * cm])
side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
story += [
    KeepTogether([
        Paragraph("5. Initial Results", h2),
        table(rows, [4.0, 6.0, 6.8], bold_last=True),
        Paragraph("Table: test accuracy / macro-F1 (%). Chance level is 20.0% on RAVDESS (always predicting "
                  "<i>neutral</i>) and 14.3% on TESS.", cap),
    ]),
    Spacer(1, 4),
    KeepTogether([side, Paragraph("Figure 3. Left: ensemble confusion matrix on the RAVDESS test set. Right: "
                                  "accuracy of every model on RAVDESS and on the unseen TESS dataset.", cap)]),
    Spacer(1, 3),
    Paragraph("<b>Working model, end to end.</b> " + code("predict_emotion") + " on six raw RAVDESS test files "
              "(true → predicted, confidence): " + "; ".join(DEMO) + ". Five of six are correct, and the wrong "
              "one is predicted with ≈100% confidence, so the model can be confidently wrong. The features it "
              "computes from a raw file are identical to the Phase 2 cache (maximum difference 0.0).", body),
    bullets(OBSERVATIONS),
    Paragraph("<b>Further experiments.</b> The same model code is also run under stricter protocols by "
              + code("scripts/run_experiments.py") + " (about 20 more training runs, so not repeated in this "
              "notebook). With <b>unseen speakers</b> (4-fold, 6 held-out actors per fold) the ensemble reaches "
              "58.5% ± 1.9, and it is clearly better on female than on male voices (66.2% vs 50.7%, p &lt; 0.001). "
              "If augmentation is done <i>before</i> the split, accuracy jumps to 96.5%, close to the paper's "
              "97.57%, which shows how the paper's number can be inflated by leakage.", body),
]

# 6. Challenges
story += [
    KeepTogether([Paragraph("6. Challenges Faced", h2), bullets([
        "<b>Details missing from the paper.</b> It does not say whether augmentation happens before or after the "
        "split, how the ensemble is trained, or where RAVDESS <i>calm</i> goes. We split by clip first (no "
        "leakage), train the two members separately and average them, and merge calm into neutral (this matches "
        "the 288 neutral clips in the paper's Table 1).",
        "<b>Inconsistent parameter table.</b> The paper lists 6,176 parameters for the last convolution, which "
        "does not fit the 128-channel Bi-LSTM output before it. We followed the layer description, so our "
        f"CNN-BiLSTM has {TRAIN['CNN-BiLSTM'][0]:,} parameters instead of 672,423.",
        "<b>Training time on a CPU.</b> Convolving a 3,672-step input with 128 filters is slow without a GPU: on "
        "our 2-core laptop (Intel i7-6600U) one epoch takes about 3.7 minutes, so training the two deep models "
        f"took {(t1['CNN'][3] + t1['CNN-BiLSTM'][3]) / 60:.1f} hours. We therefore train once in the notebook and "
        "run the many-fold experiments separately with the experiment script.",
        "<b>Avoiding inflated results.</b> Augmenting before splitting would put copies of test clips into "
        "training, and normalising with all data would leak test statistics. Both are prevented in the code. "
        "<i>Neutral</i> is 20% of RAVDESS, so we report macro-F1 next to accuracy.",
        "<b>Reproducibility.</b> All seeds are fixed, but the hardware and library versions still change the "
        "trained models: an earlier run of the same model code on a faster machine with newer NumPy / "
        "scikit-learn gave the ensemble 81.9% and the Random Forest 69.4%, against 77.1% and 67.4% here (the SVM "
        "gave 77.1% both times). All numbers in this report come from one documented run of the notebook.",
    ])]),
    Paragraph("<b>To reproduce:</b> run Phase 2 (data and feature cache), then " + code("phase3/F26-11.ipynb")
              + " top to bottom (about 3 hours on our 2-core laptop CPU). Trained models are saved in "
              + code("phase3/models/") + "; " + code("phase3/HOW_IT_WORKS.md") + " explains every function.", body),
    Paragraph("<b>References.</b> J. H. Chowdhury, S. Ramanna, K. Kotecha, “Speech emotion recognition with light "
              "weight deep neural ensemble model using hand crafted features,” <i>Sci. Rep.</i> 15, 11824 (2025). "
              "S. R. Livingstone, F. A. Russo, “RAVDESS,” <i>PLoS ONE</i> 13(5), 2018. M. K. Pichora-Fuller, "
              "K. Dupuis, “TESS,” Borealis, 2020. TensorFlow / Keras, scikit-learn, librosa documentation.", body),
]

SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                  bottomMargin=1.6 * cm, title="F26-11 Phase 3 – Model Implementation & Initial Results",
                  author="F26-11 (22L-7786, 23L-3002, 23L-3102)").build(story)
print("wrote", OUT)
