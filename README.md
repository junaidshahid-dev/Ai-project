# F26-11 — Lightweight Speech Emotion Recognition: Replication, Generalization and Bias Analysis

AI2002 Artificial Intelligence (BS-SE-5A), FAST-NUCES Lahore, Fall 2026 — **Track B (Research & Development)**
Members: 22L-7786 (Lead), 23L-3002, 23L-3102 · Instructor: Hajra Waheed

**Final report:** [`phase6/F26-11.pdf`](phase6/F26-11.pdf) · **Demo video:** [`phase6/F26-11_demo_video.mp4`](phase6/F26-11_demo_video.mp4) (6 min, narrated) ·
**Live web app:** https://f26-11-speech-emotion.streamlit.app (upload a .wav or pick an example clip)

Base paper: J. H. Chowdhury, S. Ramanna, K. Kotecha, "Speech emotion recognition with light weight deep
neural ensemble model using hand crafted features," *Scientific Reports* 15, 11824 (2025).
https://doi.org/10.1038/s41598-025-95734-z

We re-implement the paper's pipeline (ZCR, RMS energy, MFCC and Chroma features; 1D-CNN, CNN-BiLSTM and an
averaging ensemble) and test how far its reported 97.57% accuracy carries over to stricter settings: a leak-free
split, unseen speakers, an unseen dataset, and male vs. female speakers. We then explain the results with
follow-up experiments (speaker information in the features, speaker normalisation, split variance, time shift).

## Key results

| Protocol (ensemble of 1D-CNN + CNN-BiLSTM) | Accuracy |
|---|---|
| Reported in the paper (RAVDESS) | 97.57% |
| Augmentation before the train/test split (leaky, like the paper) | 96.5% |
| Clean random split, augmentation on training clips only | 81.9% |
| Unseen speakers (4-fold, 6 new actors per fold) | 58.5% ± 1.9 |
| Unseen dataset: RAVDESS → TESS / TESS → RAVDESS | 19.7% / 14.1% (chance 14.3% / 20.0%) |
| Gender gap for unseen speakers | female 66.2% vs male 50.7%, p < 0.001 |

## Repository layout

| Path | Contents |
|---|---|
| `phase1/` | Deliverable 1: project proposal |
| `phase2/` | Deliverable 2 (data): `F26-11.ipynb` / `.py` (collection, quality checks, preprocessing, EDA), report `F26-11.pdf` |
| `phase3/` | Deliverable 3 (models): `F26-11.ipynb` / `.py` (train SVM, Random Forest, CNN, CNN-BiLSTM, ensemble; predict a .wav file), report, trained models, study guide `HOW_IT_WORKS.md` |
| `phase4/` | Deliverable 4 (evaluation): `F26-11.ipynb` / `.py` (metrics with confidence intervals, model comparison, error analysis, gender-bias check), report `F26-11.pdf` |
| `phase5/` | Deliverable 5 (analysis): `F26-11.ipynb` / `.py` (reproducibility check, theoretical insights and follow-up experiments), report `F26-11.pdf` |
| `phase6/` | Deliverable 6 (final): final technical report `F26-11.pdf`, architecture figure, demo video script `VIDEO_SCRIPT.md`, study guide `HOW_IT_WORKS.md` |
| `src/ser/` | Shared code: dataset indexing (`data.py`), features and augmentation (`features.py`), models (`models.py`) |
| `scripts/` | `extract_features.py` (feature cache), `run_experiments.py` (all training runs), `analyze.py` (tables and figures), `tess_offset_check.py`, `make_report_tables.py` |
| `results/` | Saved per-clip predictions of every experiment, training histories, tables, figures, `summary.json` |
| `models/` | Final trained models used by the app: `CNN.keras`, `CNN_BiLSTM.keras`, `normalizer.npz` (trained on all of RAVDESS) |
| `app/app.py`, `app/examples/` | Streamlit demo: upload a .wav file (or pick one of 5 example clips) and get the predicted emotion |
| `report/` | Earlier LaTeX paper-style write-up of the experiments |

## Setup

Tested with Python 3.13 on Windows 10 (CPU only, no GPU needed).

```bash
git clone https://github.com/junaidshahid-dev/Ai-project.git
cd Ai-project
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow (about 1.5 GB) is only needed to train the deep models and to run the demo app. Phases 4 and 5 run
without it.

**Data.** Download both datasets from Kaggle and place them as shown (any nesting works; duplicate files are
removed automatically):

- RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio → `data/RAVDESS/`
- TESS: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess → `data/TESS/`

Then build the feature cache once (about 10 minutes):

```bash
python scripts/extract_features.py
```

## Usage

**Predict the emotion of your own recording (demo app):**

```bash
streamlit run app/app.py
```

A browser tab opens; upload a `.wav` file or pick an example clip to see the predicted emotion and the probability of each of the 7
emotions. The app uses the models in `models/`.

**Run a phase notebook:** open it in Jupyter (`jupyter notebook phase4/F26-11.ipynb`) and choose *Run All*.

| Notebook | Runtime on a 2-core laptop CPU |
|---|---|
| `phase3/F26-11.ipynb` (trains the deep models) | about 3 h |
| `phase4/F26-11.ipynb` (evaluation) | under 1 min |
| `phase5/F26-11.ipynb` (analysis experiments) | about 10 min |

## Reproduce all results

```bash
python scripts/extract_features.py                        # feature cache
python scripts/run_experiments.py paper leaky speaker cross ablation ablation_control   # ~4 h on 4 CPU cores
python scripts/tess_offset_check.py                       # zero-padding check (uses models/)
python scripts/analyze.py                                 # tables, figures, results/summary.json
```

All randomness is seeded (seed 42). Phase 5 shows that re-training the SVM from the feature cache gives the same
predictions as the saved run (100% of clips on the random split, 98.7% for unseen speakers); the Random Forest and
the deep models can differ by a few points between machines and library versions.

## Experiments

| ID | Protocol | Question |
|---|---|---|
| paper | Random 80/10/10 split, augmentation on training clips only | Can the paper's result be reproduced? |
| leaky | Augment first, then split (copies of a test clip can be in training) | Does the split order inflate accuracy? |
| speaker | 4-fold speaker-independent CV (6 held-out actors per fold) | Does it work for unseen speakers? + gender bias check |
| cross | Train RAVDESS → test TESS, and the reverse | Does it work on an unseen dataset? |
| ablation | Remove one feature group / MFCC only / no augmentation | Which features matter? |

## Citation of external resources

Datasets: RAVDESS (Livingstone & Russo, 2018, CC BY-NC-SA 4.0) and TESS (Pichora-Fuller & Dupuis, 2020).
Libraries: librosa, scikit-learn, TensorFlow/Keras, NumPy, pandas, SciPy, Matplotlib, Streamlit, ReportLab.
The model architecture and hyper-parameters follow Tables 2–3 of the base paper; all code in this repository was
written for this project.
