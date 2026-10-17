"""Build the Phase 2 report phase2/F26-11.pdf (numbers come from the executed notebook F26-11.ipynb)."""
from pathlib import Path

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

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.5, leading=13.2, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=14.5, leading=18, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, alignment=1, fontSize=10, textColor=colors.HexColor("#333333"))
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12, spaceBefore=6, spaceAfter=3)
cell = ParagraphStyle("c", parent=body, fontSize=9.5, leading=11.5, spaceAfter=0)
cap = ParagraphStyle("cap", parent=body, fontSize=9, leading=11, alignment=1, textColor=colors.HexColor("#333333"))


def bullets(items, style=body):
    return ListFlowable([ListItem(Paragraph(t, style), leftIndent=12) for t in items],
                        bulletType="bullet", start="•", leftIndent=14)


def numbered(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=14) for t in items],
                        bulletType="1", leftIndent=16)


def table(rows, widths, header=True):
    t = Table([[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=[w * cm for w in widths])
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")))
    t.setStyle(TableStyle(style))
    return t


def figure(name, width_cm, caption):
    w, h = Image.open(FIG / f"{name}.png").size
    img = RLImage(str(FIG / f"{name}.png"), width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, Paragraph(caption, cap), Spacer(1, 4)])


def link(u):
    return f'<link href="{u}" color="blue">{u}</link>'


story = [
    Paragraph("National University of Computer &amp; Emerging Sciences, Lahore", sub),
    Paragraph("AI2002 – Artificial Intelligence (BS-SE-5A) — Project Phase 2: Data Acquisition &amp; Setup", sub),
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
               + link("https://github.com/junaidshahid-dev/Ai-project"), cell), "", ""],
]
t = Table(info, colWidths=[2.2 * cm, 6.4 * cm, 2.2 * cm, 6.0 * cm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("SPAN", (1, 2), (3, 2)),
                       ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
                       ("BACKGROUND", (2, 0), (2, 1), colors.HexColor("#eeeeee"))]))
story += [t, Spacer(1, 6)]

# 1. Dataset source & description
story += [
    Paragraph("1. Dataset Source and Description", h2),
    Paragraph("We use the two acted-speech corpora used by our base paper (Chowdhury et al., <i>Scientific "
              "Reports</i> 15, 11824, 2025). Both are public, studio-recorded and labelled by design: the emotion "
              "of every clip is encoded in its file name.", body),
    table([
        ["", "RAVDESS (Livingstone &amp; Russo, 2018)", "TESS (Pichora-Fuller &amp; Dupuis, 2020)"],
        ["Source", "Kaggle uwrfkaggler/ravdess-emotional-speech-audio; official: Zenodo record 1188976",
         "Kaggle ejlok1/toronto-emotional-speech-set-tess; official: U of T Dataverse doi:10.5683/SP2/E8H2MF"],
        ["Content", "1,440 speech clips; 24 professional actors (12 male, 12 female); 2 fixed sentences; "
                    "8 emotions at 2 intensities", "2,800 clips; 2 female speakers (aged 26 and 64); carrier "
                    "phrase “Say the word __” with 200 target words; 7 emotions"],
        ["Format", "WAV, 16-bit PCM, 48 kHz, ~3.7 s per clip (563 MB)", "WAV, 16-bit PCM, 24.4 kHz, ~2.1 s per clip (268 MB)"],
        ["Licence", "CC BY-NC-SA 4.0", "Research use, per the Borealis Dataverse terms"],
        ["Role", "Training, within-dataset and speaker-independent tests, gender-bias check",
         "Unseen dataset for the cross-dataset test"],
    ], [2.0, 7.4, 7.4]),
    Spacer(1, 4),
]

# 2. Collection method
story += [
    Paragraph("2. Data Collection Method", h2),
    Paragraph("No scraping or manual labelling is needed: we download the raw <font name='Courier'>.wav</font> "
              "files from a public repository and decode labels from file names "
              "(RAVDESS <font name='Courier'>03-01-05-02-01-01-12.wav</font> = audio-only, speech, <b>emotion</b>, "
              "intensity, statement, repetition, <b>actor</b>, odd actor = male; TESS "
              "<font name='Courier'>OAF_back_angry.wav</font> = <b>speaker</b>, word, <b>emotion</b>). "
              "The official servers were throttled to about 35 KB/s on our connection, so we downloaded the same "
              "files from public Hugging Face mirrors (<font name='Courier'>birgermoell/ravdess</font>, "
              "<font name='Courier'>myleslinder/tess</font>) with the download cell of the notebook. "
              "<b>Verification:</b> we compared every RAVDESS file with the CRC32 checksums inside the official "
              "Zenodo archive: all 1,440 are bit-identical. TESS gives the expected 2,800 files (400 per emotion, "
              "1,400 per speaker). Kaggle downloads placed in <font name='Courier'>data/</font> work the same way.", body),
]

# 3. Preprocessing
story += [
    Paragraph("3. Preprocessing Steps", h2),
    Paragraph("<b>3.1 Cleaning and label mapping.</b> We ran automatic checks on all 4,240 files: no unreadable, "
              "empty, silent or unlabelled files and no duplicate file names. The checks did find three anomalies "
              "in the official releases, all handled:", body),
    bullets([
        "<b>5 stereo RAVDESS files</b> (actors 01, 05, 20) → converted to mono by averaging the channels.",
        "<b>Mixed sampling rates</b> (48 kHz, 24,414 Hz and one TESS file, OAF_food_fear.wav, at 96 kHz) → "
        "everything is resampled to 22,050 Hz.",
        "<b>One duplicated RAVDESS recording</b> (actor 07, <i>happy</i>, repetitions 1 and 2 are the same audio). "
        "We keep both to match the paper's 1,440 clips and tracked the pair: in the random split one copy is in "
        "training and one in validation, so no test clip is affected.",
        "<b>Common label set:</b> RAVDESS has 8 emotions and TESS 7. As in the paper (288 neutral clips in its "
        "Table 1), <i>calm</i> is merged into <i>neutral</i>; TESS <i>fear</i> → fearful, <i>ps</i> (pleasant "
        "surprise) → surprised. Result: 7 classes in both datasets.",
    ]),
    Paragraph("<b>3.2 Feature pipeline (as in the base paper).</b>", body),
    numbered([
        "Load each clip as mono, resample to 22,050 Hz and read <b>2.5 s from an offset of 0.6 s</b> "
        "(skips the leading silence).",
        "Frame length 2,048 and hop 512 give <b>108 frames</b>; clips shorter than the window are zero-filled.",
        "Per frame: zero-crossing rate (1), RMS energy (1), 20 MFCCs and 12 Chroma-STFT bins → a "
        "<b>108 × 34 matrix</b>, flattened to the paper's <b>3,672-value</b> input vector.",
        "<b>Augmentation</b> (training clips only): Gaussian noise (α = 0.035 × peak), pitch shift (+0.7 "
        "semitone) and noise + pitch → 4 versions per clip, seeded per file for reproducibility.",
        "<b>Standardisation</b> (z-score per dimension) with mean/std from the <i>training split only</i>.",
    ]),
    Paragraph("<b>Leakage control:</b> splits are made on clips (or speakers) <i>before</i> augmentation, so all "
              "versions of a clip stay on one side. <b>Change from the Phase 1 proposal:</b> we dropped silence "
              "trimming and time-stretch augmentation to reproduce the paper's pipeline exactly; the fixed 0.6 s "
              "offset already removes leading silence and the paper uses only noise and pitch augmentation.", body),
]

# 4. EDA
story += [
    Paragraph("4. Exploratory Data Analysis — Key Insights", h2),
    figure("fig1_class_balance", 14.5, "Figure 1. Clips per emotion after label mapping."),
    bullets([
        "<b>Balance.</b> TESS has exactly 400 clips per emotion. RAVDESS has 192 per emotion except <i>neutral</i> "
        "(288 = 96 neutral + 192 calm, 20% of the data), so we report macro-F1 next to accuracy.",
        "<b>Gender.</b> RAVDESS has 720 male and 720 female clips with identical emotion and intensity counts, so "
        "the gender-bias check is fair. TESS has only two female voices and is not used for it.",
    ]),
    figure("fig2_durations", 11, "Figure 2. Clip durations; the dashed line marks the end of the paper's "
           "0.6 s + 2.5 s window."),
    bullets([
        "<b>Clip length differs between datasets.</b> RAVDESS clips average 3.70 s, TESS 2.06 s. All TESS clips "
        "(100%) are shorter than the 3.1 s window, so 41.5% of TESS feature frames are zero padding, against 0.02% "
        "for RAVDESS (only 15 of its 1,440 clips need a few padded frames). "
        "A RAVDESS-trained model never sees this padding, which is an early warning for the cross-dataset test.",
        "<b>Arousal is visible in simple features</b> (Fig. 3, notebook Sec. 6). Median RMS energy of <i>angry</i> "
        "clips is about 8.5× that of <i>neutral</i> (0.029 vs 0.0034); happy, fearful and surprised are also loud, "
        "neutral and sad are quiet. Zero-crossing rate on its own separates emotions poorly, while MFCC profiles "
        "give each emotion a distinct pattern.",
    ]),
    figure("fig5_energy_zcr", 15, "Figure 3. RAVDESS mean RMS energy and zero-crossing rate per emotion "
           "(voiced frames, original clips)."),
    bullets([
        "<b>Strong dataset shift.</b> In a PCA of clip-level features (Fig. 4) RAVDESS and TESS form two largely "
        "separate clusters, more clearly than the emotions do, and the MFCC pattern of the same emotion differs "
        "between the corpora. We therefore expect within-dataset accuracy to overstate how well the model "
        "generalises, which is the hypothesis Phase 3 tests.",
    ]),
    figure("fig7_pca_shift", 14, "Figure 4. PCA of clip-level features (mean and std of the 34 features), coloured "
           "by dataset (left) and emotion (right)."),
]

# 5. Final dataset summary
story += [
    Paragraph("5. Final Dataset Summary", h2),
    table([
        ["", "RAVDESS", "TESS"],
        ["Clips / speakers", "1,440 / 24 (12 M, 12 F)", "2,800 / 2 (F)"],
        ["Classes", "7 (neutral 288, others 192 each)", "7 (400 each)"],
        ["Per-clip representation", "108 frames × 34 features → 3,672 values", "108 frames × 34 features → 3,672 values"],
        ["Cached feature array", "(1,440, 4, 108, 34), float32, 81 MB", "(2,800, 4, 108, 34), float32, 157 MB"],
        ["Missing / NaN values", "0 (0.02% of frames zero-filled)", "0 (41.5% of frames zero-filled)"],
    ], [4.0, 6.4, 6.4]),
    Spacer(1, 5),
    Paragraph("<b>Splits</b> (seed 42, stratified by emotion; validation is used only for early stopping):", body),
    table([
        ["Protocol", "Train clips (×4 aug.)", "Val clips", "Test clips"],
        ["P1 random 80/10/10, as in the paper", "1,152 (4,608)", "144", "144"],
        ["P2 speaker-independent, 4 folds of 6 actors (3 M + 3 F held out)", "972 (3,888) per fold", "108", "360"],
        ["P3 cross-dataset: train RAVDESS, test all of TESS (and reverse)", "1,296 (5,184)", "144", "2,800"],
    ], [8.2, 3.6, 2.0, 3.0]),
    Spacer(1, 5),
    Paragraph("<b>Outputs:</b> <font name='Courier'>features/ravdess.npz</font>, "
              "<font name='Courier'>features/tess.npz</font> and their metadata CSVs (label, speaker, gender, "
              "intensity), plus <font name='Courier'>phase2/dataset_index.csv</font> (one row per clip with "
              "duration, sampling rate and channels). The data is ready for model implementation in Phase 3. "
              "To reproduce: <font name='Courier'>pip install -r requirements.txt</font>, then run "
              "<font name='Courier'>phase2/F26-11.ipynb</font> top to bottom (it downloads the data if missing and "
              "runs <font name='Courier'>scripts/extract_features.py</font>, about 7 minutes on 4 CPU cores).", body),
    Paragraph("<b>References.</b> J. H. Chowdhury, S. Ramanna, K. Kotecha, “Speech emotion recognition with light "
              "weight deep neural ensemble model using hand crafted features,” <i>Sci. Rep.</i> 15, 11824 (2025). "
              "S. R. Livingstone, F. A. Russo, “The Ryerson Audio-Visual Database of Emotional Speech and Song "
              "(RAVDESS),” <i>PLoS ONE</i> 13(5), 2018. M. K. Pichora-Fuller, K. Dupuis, “Toronto emotional speech "
              "set (TESS),” Borealis, V1, 2020. B. McFee et al., “librosa,” <i>Proc. SciPy</i>, 2015.", body),
]

SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                  bottomMargin=1.6 * cm, title="F26-11 Phase 2 – Data Acquisition & Setup",
                  author="F26-11 (22L-7786, 23L-3002, 23L-3102)").build(story)
print("wrote", OUT)
