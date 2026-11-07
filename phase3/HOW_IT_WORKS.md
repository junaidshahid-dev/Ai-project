# Phase 3 — how the code works (study guide)

This guide walks through `F26-11.ipynb` (same code as `F26-11.py`) in the order it runs, in plain words,
so every group member can explain it. Read it next to the notebook.

## The idea in one paragraph

We want a model that hears a short speech clip and says which of 7 emotions it is (neutral, happy, sad,
angry, fearful, disgust, surprised). In Phase 2 we already turned every clip into numbers: 108 time frames,
and for each frame 34 values (ZCR, RMS energy, 20 MFCCs, 12 Chroma). In Phase 3 we **train models on these
numbers**, **test them**, and **use them on a new .wav file**.

## Step by step

| Step | Code | What happens |
|---|---|---|
| 0 | imports, `ROOT`, `SEED = 42` | Load the libraries. `SEED` makes every random choice the same each run, so results can be repeated. |
| 1 | `load_features(name)` | Opens the Phase 2 files `features/ravdess.npz` (the numbers) and `features/ravdess.csv` (the labels). RAVDESS array shape is `(1440, 4, 108, 34)` = 1,440 clips × 4 versions (original, noise, pitch, noise+pitch) × 108 frames × 34 features. |
| 2 | `random_split(y)` | Splits the 1,440 clip numbers into 80% train (1,152), 10% validation (144), 10% test (144). `stratify` keeps the same share of each emotion in every part. |
| 3 | `get_frames(X, y, idx, augment)` | Picks the chosen clips. For training (`augment=True`) it takes all 4 versions, so 1,152 clips become 4,608 samples. For validation/test it takes only the original version. |
| 3 | `to_vectors(frames)` | Turns each 108 × 34 matrix into one long row of 3,672 numbers (all 108 ZCR values, then 108 RMS values, …), the input format of the paper. |
| 3 | `to_summary(frames)` | For the simple models: the mean and standard deviation of each of the 34 features over time → 68 numbers per clip. |
| 4 | `conv_block(...)` | One building block of the network: Conv1D (finds patterns) → BatchNormalization (keeps values in a stable range) → MaxPooling (halves the length) → optional Dropout (randomly switches off 20% of values during training to reduce overfitting). |
| 4 | `build_model(use_bilstm)` | Stacks 5 blocks (128, 128, 64, 64, 32 filters), then Flatten → Dense(128) → Dense(7, softmax). Softmax gives 7 probabilities that add up to 1. With `use_bilstm=True` a Bidirectional LSTM is added before the last block; it reads the sequence forwards and backwards. Copied from Table 2 of the base paper. |
| 4 | `train_model(...)` | Trains with Adam (learning rate 0.001), batches of 64, up to 100 epochs. **EarlyStopping** stops when validation accuracy has not improved for 5 epochs and keeps the best weights. **ReduceLROnPlateau** halves the learning rate if validation accuracy stalls for 3 epochs. `[..., None]` adds the "channel" dimension Conv1D expects: `(n, 3672)` → `(n, 3672, 1)`. |
| 5 | `train_all(X, y, train, val)` | Does the whole training on one split: builds the inputs, computes the **mean and std of the training data only** (z-score normalisation), trains the SVM and Random Forest, then the CNN and CNN-BiLSTM. Returns everything in one dictionary. |
| 5 | `predict_all(trained, frames)` | Gives the class probabilities of every model. **Ensemble** = average of the CNN and CNN-BiLSTM probabilities. |
| 5 | `evaluate(probs, y_true)` | The predicted class is the one with the highest probability (`argmax`). Computes accuracy and macro-F1 for each model. |
| 6 | Experiment 1 | Train on RAVDESS train set, test on the 144 RAVDESS test clips. Prints the results table, a per-emotion report, training curves and a confusion matrix. |
| 7 | Experiment 2 | The same trained models are tested on all 2,800 TESS clips, a dataset they never saw. This checks if the model generalises. |
| 8 | summary, bar chart, save | Saves `results.csv`, the figures and the trained models (`models/CNN.keras`, `models/CNN-BiLSTM.keras`, `models/normalizer.npz`). |
| 9 | `extract_features(path)` | The full input → output chain for a new .wav file: load 2.5 s starting at 0.6 s, compute the 34 features for 108 frames, fill missing frames with zeros. The last line checks it gives exactly the same numbers as the Phase 2 cache. |
| 9 | `predict_emotion(path, trained)` | Features → normalise → models → ensemble → returns the emotion and how confident the model is. |

## Our results (from the executed notebook)

| Model | RAVDESS test (144 clips): accuracy / macro-F1 | TESS, unseen (2,800 clips): accuracy / macro-F1 |
|---|---|---|
| SVM | 77.1% / 76.7% | 22.4% / 15.6% |
| Random Forest | 67.4% / 66.0% | 30.8% / 25.3% |
| CNN | 76.4% / 76.5% | 23.0% / 20.8% |
| CNN-BiLSTM | 77.1% / 75.9% | 17.1% / 18.5% |
| Ensemble | 77.1% / 75.9% | 21.0% / 19.9% |

- Chance level: 20% on RAVDESS (always saying "neutral"), 14.3% on TESS (1 of 7).
- CNN trained 18 epochs (best validation 75.7% at epoch 13, 72 min); CNN-BiLSTM 21 epochs (best 79.9% at epoch 16, 95 min).
- Ensemble on RAVDESS: 111 of 144 correct. Neutral 29/29; angry and disgust 16/19; happy and surprised 12/19 (hardest).
- `predict_emotion` on 6 raw test files: 5 correct.

## Words you may be asked about

- **Epoch**: one pass over all 4,608 training samples.
- **Batch size 64**: the model updates its weights after every 64 samples.
- **Validation set**: not used for learning the weights, only to decide when to stop and when to lower the learning rate.
- **Test set**: used once at the end to measure the final result.
- **Accuracy**: correct predictions ÷ all predictions.
- **Macro-F1**: F1 score for each emotion, then averaged, so every emotion counts equally (neutral has more clips than the others).
- **Confusion matrix**: rows = true emotion, columns = predicted emotion; the diagonal is correct.
- **Overfitting**: training accuracy keeps rising but validation accuracy stops; dropout and early stopping fight this.
- **Data leakage**: test information getting into training. We avoid it by splitting clips *before* using augmented copies, and by normalising with training statistics only.

## Likely questions and short answers

1. **Why these models?** We are testing a published paper (Track B), so we use its 1D-CNN, CNN-BiLSTM and averaging ensemble with its settings. SVM and Random Forest are simple baselines to check whether deep learning actually helps.
2. **Why augment only the training clips?** If noisy or pitch-shifted copies of a test clip were in training, the model would have almost seen the test clip, and accuracy would look too high.
3. **Why z-score with training mean/std only?** Using test data to compute them would leak test information.
4. **Why is the TESS result so low?** Different speakers, recording setup and clip length (TESS clips are short, so many frames are zero padding). The model learned RAVDESS-specific patterns, not general emotion cues.
5. **Why is the paper's 97.57% higher than ours?** Our split keeps every version of a test clip out of training. When we augment *before* splitting (in `scripts/run_experiments.py`), we also get about 96.5%, so the paper's number is probably inflated by this leakage.
6. **Why does the ensemble not beat the SVM here?** The test set has only 144 clips, so one clip = 0.7 points; differences of a few points are within run-to-run variation. In an earlier run of the same code the ensemble reached 81.9%. The bigger lesson is that a simple SVM on 68 summary values is already a strong baseline.
7. **Why does training accuracy reach ~100% but validation only ~75–80%?** The model overfits (memorises the training samples). Dropout, batch normalisation and early stopping limit this.
8. **How long does training take?** About 3.7 minutes per epoch on our 2-core laptop CPU (no GPU): 72 min for the CNN and 95 min for the CNN-BiLSTM.
