# F26-11 — Lightweight Speech Emotion Recognition: Replication, Generalization and Bias Analysis

AI2002 Artificial Intelligence (BS-SE-5A), FAST-NUCES Lahore, Fall 2026 — **Track B (Research & Development)**
Members: 22L-7786 (Lead), 23L-3002 · Instructor: Hajra Waheed

Base paper: J. H. Chowdhury, S. Ramanna, K. Kotecha, "Speech emotion recognition with light weight deep
neural ensemble model using hand crafted features," *Scientific Reports* 15, 11824 (2025).
https://doi.org/10.1038/s41598-025-95734-z

We re-implement the paper's pipeline (ZCR, RMSE, MFCC and Chroma STFT features; 1D-CNN, CNN_Bi-LSTM and an
averaging ensemble) and test how far its near-perfect accuracy carries over to stricter settings: an
augmentation-safe split, unseen speakers, an unseen dataset, and male vs. female speakers.

## Repository layout

| Path | Contents |
|---|---|
| `src/ser/data.py` | Dataset indexing and the common 7-emotion label set |
| `src/ser/features.py` | Paper feature pipeline and augmentation |
| `src/ser/models.py` | Baselines (Majority, SVM, Random Forest) and the paper's CNN / CNN_Bi-LSTM / ensemble |
| `scripts/extract_features.py` | Builds the feature cache (`features/`) |
| `scripts/run_experiments.py` | Runs all experiments, writes `results/predictions_*.csv` |
| `scripts/tess_offset_check.py` | Controlled test of the zero-padding explanation (Sec. 4.4 of the report) |
| `scripts/analyze.py` | Metrics, bias check and figures (`results/tables`, `results/figures`) |
| `scripts/make_report_tables.py` | Writes every number and table used in the report (`report/generated/`) |
| `models/` | Final RAVDESS-trained CNN and CNN_Bi-LSTM used by the demo app |
| `app/app.py` | Optional Streamlit demo (bonus) |
| `report/` | LaTeX technical report (`main.tex`) and compiled PDF (`F26-11_Final_Report.pdf`) |
| `phase1/` | Phase 1 proposal |

## Data

Both datasets are on Kaggle. Download them and place the folders as shown (any nesting works; duplicate
files are removed automatically):

- RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio → `data/RAVDESS/`
- TESS: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess → `data/TESS/`

## Reproduce

```bash
pip install -r requirements.txt
python scripts/extract_features.py                       # ~10 min on 4 CPU cores
python scripts/run_experiments.py paper leaky speaker cross ablation ablation_control   # ~4 h on a 4-core CPU
python scripts/tess_offset_check.py                      # controlled zero-padding check (uses models/)
python scripts/analyze.py                                # tables + figures + results/summary.json
python scripts/make_report_tables.py                     # LaTeX numbers and tables for the report
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
streamlit run app/app.py                                 # optional demo
```

All randomness is seeded (seed 42). Results in the report were produced on CPU with the versions in
`requirements.txt`; GPU runs can differ slightly because of non-deterministic kernels.

## Experiments

| ID | Protocol | Question |
|---|---|---|
| paper | Random 80/10/10 split, augmentation on training clips only | Can the paper's result be reproduced? |
| leaky | Augment first, then split (copies of a test clip can be in training) | Does the split order inflate accuracy? |
| speaker | 4-fold speaker-independent CV (6 held-out actors per fold) | Does it work for unseen speakers? + gender bias check |
| cross | Train RAVDESS → test TESS, and reverse | Does it work on an unseen dataset? |
| ablation | Remove one feature group / MFCC only / no augmentation (+ patience control) | Which features matter? |

## Citation of external resources

Datasets: RAVDESS (Livingstone & Russo, 2018, CC BY-NC-SA 4.0) and TESS (Pichora-Fuller & Dupuis, 2020).
Libraries: librosa, scikit-learn, TensorFlow/Keras, NumPy, pandas, Matplotlib. The model architecture and
hyper-parameters follow Tables 2–3 of the base paper; all code in this repository was written for this project.

## Key results (see the report for details)

| Protocol | Ensemble accuracy |
|---|---|
| Reported in the paper (RAVDESS) | 97.57% |
| Paper protocol, augmentation after split | 81.9% |
| Augmentation before split (leaky) | 96.5% |
| Speaker-independent (4-fold) | 58.5% ± 1.9 |
| RAVDESS → TESS / TESS → RAVDESS | 19.7% / 14.1% (chance 14.3% / 20.0%) |
| Gender gap (speaker-independent) | female 66.2% vs male 50.7%, p < 0.001 |
