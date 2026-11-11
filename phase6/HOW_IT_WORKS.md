# Study guide for Deliverables 4, 5 and 6 (presentation and viva)

Read this together with `phase3/HOW_IT_WORKS.md` (how the models are built and trained).

## The whole project in 30 seconds

A 2025 paper says its small CNN ensemble recognises 7 emotions from speech with 97.57% accuracy. We rebuilt it
and tested it fairly. Result: 81.9% on a clean split, 96.5% only when test clips leak into training, 58.5% for
new speakers, near chance on a new dataset, and 15.6 points worse for male voices. Phase 5 shows why: the
features describe *who* is speaking more than *how they feel*.

## Deliverable 4 — `phase4/F26-11.ipynb` (evaluation)

| Function / step | What it does |
|---|---|
| `load_predictions()` | Reads `results/predictions_*.csv`: one row per test clip, model and protocol (true label, predicted label, speaker, gender). |
| `scores(df)` | Accuracy, macro-F1, weighted F1, precision and recall. |
| `bootstrap_ci(correct)` | Picks the test clips randomly with replacement 2,000 times and recomputes accuracy each time; the middle 95% of those values is the confidence interval. |
| check against `summary.json` | Proves the notebook's numbers equal the saved experiment summary. |
| `mcnemar(df, a, b)` | Compares two models on the same clips: counts clips only A gets right and only B gets right; if the models were equal these would be 50/50 (binomial test). |
| `recall_per_emotion(df)` | Share of each emotion's clips recognised correctly. |
| `two_proportion_test(a, b)` | Are two accuracies (e.g. male vs female) really different? z-test. |
| Mann–Whitney test | Same question at actor level (12 male vs 12 female per-actor accuracies), because clips of one actor are not independent. |

## Deliverable 5 — `phase5/F26-11.ipynb` (why it happens)

| Function / step | What it does |
|---|---|
| `fit_predict(...)` | Trains a model on the augmented training clips and predicts the original test clips. |
| `compare_with_saved(...)` | Re-trains the SVM and checks how many predictions equal the saved ones (100% on the random split → reproducible). |
| `eta_squared(values, groups)` | For each feature, the share of its variation explained by the group (speaker or emotion). Speaker 27.9% vs emotion 7.6%. |
| speaker-ID SVM | Same features, but predict the actor: 95.0% → the features are a "voice fingerprint". |
| `speaker_normalise(X, meta)` | Z-scores each speaker's features with that speaker's own mean and std (no labels needed). SVM: 49.3% → 59.2% for new speakers, 24.2% → 34.8% on TESS; the gender gap does not close. |
| 10 random splits | Same SVM gets 72.9%–83.3% just by changing the split → a 144-clip test set is noisy. |
| `shift(frames, k)` | Delays the speech by k frames. The flattened 3,672-value input loses 15.3 points at 0.12 s; the 68-value summary only 2.1. |

## Deliverable 6 — final submission

* Report: `phase6/F26-11.pdf` (8 pages: abstract, method and architecture diagrams, implementation steps,
  results, replication vs paper, error analysis, bias, ablation, theory, demo, challenges, how to run).
* Code and models: the GitHub repository (README has setup and usage). Models: `models/CNN.keras`,
  `models/CNN_BiLSTM.keras`, `models/normalizer.npz`.
* Demo: `streamlit run app/app.py` → upload a .wav → predicted emotion and 7 probabilities.
* Video: follow `phase6/VIDEO_SCRIPT.md`.

## Likely questions

1. **Why is your accuracy lower than the paper's?** We split the clips before augmenting, so no copy of a test
   clip is in training. If we augment first (like the paper probably did) we also get 96.5%.
2. **What is data leakage?** Information about the test set getting into training. Here: a noisy copy of a test
   recording in the training set (99.1% of test samples had one).
3. **Why is "unseen speakers" the most important result?** A real user is always a new speaker; 58.5% is the
   realistic accuracy.
4. **Is the ensemble better than the SVM?** On 144 clips the difference (4.9 points) is not significant
   (McNemar p = 0.30). On 1,440 unseen-speaker clips it is (+9.4 points, p < 0.001).
5. **What is a confidence interval?** The range where the true accuracy probably lies; with 144 clips it is
   about ±6 points.
6. **Why does the model fail on TESS?** Different speakers, single words and recording setup, and 41.5% of TESS
   frames are zero padding (short clips), which the model reads as quiet "sad" speech; it predicts sad for 53%.
7. **Is the model biased?** Yes: 50.7% for men vs 66.2% for women on new speakers (p < 0.001), even though
   the data have 12 male and 12 female actors.
8. **Does speaker normalisation fix the bias?** No. It helps everyone, but women more (gap 7.5 → 11.0 for the
   SVM), so the bias has another cause, probably that men's emotional changes are less visible in these features.
9. **Which features matter?** MFCC. Without it accuracy falls by 22 points; ZCR and RMS add nothing measurable.
10. **Why use the SVM in Phase 5 and not the CNN?** It trains in seconds instead of hours on our CPU and shows
    the same trends, so it is a fast way to test each explanation.
