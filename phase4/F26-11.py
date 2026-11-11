# %% [markdown]
# # F26-11 — Phase 4: Evaluation and Analysis of Results
# **Lightweight Speech Emotion Recognition Using Hand-Crafted Acoustic Features**
#
# AI2002 Artificial Intelligence (BS-SE-5A), FAST-NUCES Lahore, Fall 2026 · Track B (Research & Development)
# Members: 22L-7786 (Lead), 23L-3002, 23L-3102 · Instructor: Hajra Waheed
#
# This notebook evaluates the models built in Phase 3 (SVM, Random Forest, 1D-CNN, CNN-BiLSTM and their
# ensemble) under five test protocols, compares them, analyses where they fail and checks for gender bias.
#
# It reads the saved per-clip predictions in `results/predictions_*.csv`. They were written by
# `scripts/run_experiments.py`, which trains the same models as the Phase 3 notebook under every protocol
# (23 deep-model training runs). Evaluating saved predictions takes seconds and needs no TensorFlow.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, mannwhitneyu, norm
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)

ROOT = Path.cwd() if (Path.cwd() / "results").exists() else Path.cwd().parent
RESULTS = ROOT / "results"
OUT = ROOT / "phase4"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
MODELS = ["Majority", "SVM", "RandomForest", "CNN", "CNN_BiLSTM", "Ensemble"]
PROTOCOLS = {"paper": "Random split (paper)", "leaky": "Augment before split",
             "speaker": "Unseen speakers", "ravdess_to_tess": "RAVDESS → TESS",
             "tess_to_ravdess": "TESS → RAVDESS"}
COLORS = {"SVM": "#2a78d6", "RandomForest": "#eb6834", "CNN": "#1baf7a", "CNN_BiLSTM": "#eda100",
          "Ensemble": "#e87ba4"}
rng = np.random.default_rng(42)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "font.size": 9})

# %% [markdown]
# ## 1. Load the saved predictions
# One row per test clip, model and protocol: true label, predicted label, speaker and gender.
#
# | Protocol | Train | Test | Question |
# |---|---|---|---|
# | Random split (paper) | 80% of RAVDESS clips (+ augmented copies) | 144 RAVDESS clips | Does the model work? |
# | Augment before split | 80% of all 5,760 augmented samples | 576 samples | Does the paper's split order inflate accuracy? |
# | Unseen speakers | 18 actors, 4 folds | 6 new actors per fold (all 1,440 clips) | Does it work for new voices? |
# | RAVDESS → TESS | all of RAVDESS | 2,800 TESS clips | Does it work on a new dataset? |
# | TESS → RAVDESS | all of TESS | 1,440 RAVDESS clips | The reverse direction |

# %%
def load_predictions():
    """All saved per-clip predictions in one table, with a column saying if each prediction is right."""
    files = sorted(RESULTS.glob("predictions_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["correct"] = df["pred"] == df["label"]
    return df


preds = load_predictions()
main = preds[preds["variant"] == "full"]
print(main.groupby(["experiment", "model"]).size().unstack()[MODELS])

# %% [markdown]
# ## 2. Evaluation metrics
# * **Accuracy**: share of correct predictions.
# * **Macro-F1**: F1 of every emotion, averaged, so each emotion counts equally (RAVDESS has more neutral clips).
# * **Weighted F1**: the metric the base paper reports, used only to compare with it.
# * **95% confidence interval** of accuracy: we resample the test clips 2,000 times (bootstrap). A wide interval
#   means the test set is too small to trust small differences.

# %%
def scores(df):
    """Accuracy, macro-F1, weighted-F1, macro precision and macro recall of one set of predictions."""
    y, p = df["label"], df["pred"]
    kw = dict(labels=range(len(EMOTIONS)), zero_division=0)
    return {"n": len(df), "accuracy": accuracy_score(y, p),
            "macro_F1": f1_score(y, p, average="macro", **kw),
            "weighted_F1": f1_score(y, p, average="weighted", **kw),
            "precision": precision_score(y, p, average="macro", **kw),
            "recall": recall_score(y, p, average="macro", **kw)}


def bootstrap_ci(correct, n_boot=2000):
    """95% confidence interval of accuracy, by resampling the test clips with replacement."""
    correct = np.asarray(correct, dtype=float)
    samples = rng.choice(correct, size=(n_boot, len(correct)))
    return np.percentile(samples.mean(axis=1), [2.5, 97.5])


rows = []
for exp, name in PROTOCOLS.items():
    for model in MODELS:
        d = main[(main["experiment"] == exp) & (main["model"] == model)]
        low, high = bootstrap_ci(d["correct"])
        rows.append({"experiment": exp, "protocol": name, "model": model, **scores(d),
                     "ci_low": low, "ci_high": high})
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "metrics.csv", index=False)
print((metrics.set_index(["protocol", "model"]).drop(columns=["experiment", "n"]) * 100).round(1))

# %%
summary = json.load(open(RESULTS / "summary.json"))
for r in summary["model_comparison"]:
    mine = metrics[(metrics["experiment"] == r["experiment"]) & (metrics["model"] == r["model"])].iloc[0]
    assert abs(mine["accuracy"] - r["acc"]) < 1e-9 and abs(mine["macro_F1"] - r["f1_macro"]) < 1e-9
print("All accuracies and macro-F1 values match results/summary.json.")

speaker = main[main["experiment"] == "speaker"]
folds = speaker.groupby(["model", "fold"])["correct"].mean().unstack()
print("\nUnseen speakers, accuracy per fold (%):")
print((folds.loc[MODELS] * 100).round(1).assign(mean=lambda t: t.mean(axis=1).round(1),
                                                std=lambda t: t.iloc[:, :4].std(axis=1, ddof=0).round(1)))

# %% [markdown]
# ## 3. Comparison of models
# **McNemar's test** compares two models on the *same* test clips. It only looks at clips where exactly one of
# the two models is right; if both models were equally good, those clips would split 50/50. A small p-value
# (< 0.05) means the difference is real and not luck.

# %%
def mcnemar(df, model_a, model_b):
    """Clips only model A gets right, clips only model B gets right, and the exact McNemar p-value."""
    a = df[df["model"] == model_a]
    b = df[df["model"] == model_b]
    assert (a["path"].values == b["path"].values).all()
    a, b = a["correct"].values, b["correct"].values
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


pairs = [("Ensemble", "SVM"), ("Ensemble", "CNN"), ("CNN_BiLSTM", "CNN"), ("SVM", "RandomForest")]
rows = []
for exp, name in PROTOCOLS.items():
    d = main[main["experiment"] == exp]
    for a, b in pairs:
        only_a, only_b, p = mcnemar(d, a, b)
        rows.append({"protocol": name, "comparison": f"{a} vs {b}", "only A right": only_a,
                     "only B right": only_b, "p_value": round(p, 4)})
mcnemar_table = pd.DataFrame(rows)
mcnemar_table.to_csv(OUT / "mcnemar.csv", index=False)
print(mcnemar_table.to_string(index=False))

# %%
runs = pd.read_csv(RESULTS / "runs_paper.csv").set_index("model")
cost = runs.loc[MODELS[1:], ["params", "fit_s", "infer_ms", "epochs"]]
cost["fit_min"] = (cost["fit_s"] / 60).round(1)
print("Size and speed under the random-split protocol (4-core CPU):")
print(cost[["params", "epochs", "fit_min", "infer_ms"]].round(2))

# %%
fig, ax = plt.subplots(figsize=(8.5, 3.4))
shown = MODELS[1:]
for i, (exp, name) in enumerate(PROTOCOLS.items()):
    for j, model in enumerate(shown):
        r = metrics[(metrics["experiment"] == exp) & (metrics["model"] == model)].iloc[0]
        x = i + (j - 2) * 0.15
        ax.errorbar(x, 100 * r["accuracy"], yerr=[[100 * (r["accuracy"] - r["ci_low"])],
                    [100 * (r["ci_high"] - r["accuracy"])]], fmt="o", ms=5, lw=1.5, capsize=0,
                    color=COLORS[model], label=model if i == 0 else None)
    chance = metrics[(metrics["experiment"] == exp) & (metrics["model"] == "Majority")]["accuracy"].item()
    ax.hlines(100 * chance, i - 0.4, i + 0.4, color="#52514e", ls=":", lw=1.2,
              label="chance (majority class)" if i == 0 else None)
ax.set_xticks(range(len(PROTOCOLS)), PROTOCOLS.values())
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#e6e5e0")
ax.legend(ncol=6, fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()
plt.savefig(OUT / "figures" / "fig1_accuracy_ci.png", dpi=200)
plt.show()

# %% [markdown]
# ## 4. Error analysis — where the model does well and where it fails
# We analyse the ensemble (the paper's final model).
#
# ### 4.1 Per emotion

# %%
ens = main[main["model"] == "Ensemble"]
for exp in ["paper", "speaker"]:
    d = ens[ens["experiment"] == exp]
    print(f"--- {PROTOCOLS[exp]} ---")
    print(classification_report(d["label"], d["pred"], labels=range(7), target_names=EMOTIONS, digits=3,
                                zero_division=0))


def recall_per_emotion(df):
    """Share of the clips of each emotion that are recognised correctly."""
    return df.groupby("emotion")["correct"].mean().reindex(EMOTIONS)


recalls = pd.DataFrame({PROTOCOLS[e]: recall_per_emotion(ens[ens["experiment"] == e])
                        for e in ["paper", "speaker", "ravdess_to_tess"]})
print((100 * recalls).round(1))

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
short = ["neu", "hap", "sad", "ang", "fea", "dis", "sur"]
for ax, exp in zip(axes, ["paper", "speaker", "ravdess_to_tess"]):
    d = ens[ens["experiment"] == exp]
    cm = confusion_matrix(d["label"], d["pred"], labels=range(7), normalize="true")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(7), short)
    ax.set_yticks(range(7), short)
    ax.set_xlabel("Predicted")
    ax.set_title(f"{PROTOCOLS[exp]} (n = {len(d)})", fontsize=9)
axes[0].set_ylabel("True emotion")
plt.tight_layout()
plt.savefig(OUT / "figures" / "fig2_confusion.png", dpi=200)
plt.show()

# %% [markdown]
# ### 4.2 Most common mistakes for unseen speakers

# %%
d = ens[ens["experiment"] == "speaker"]
cm = confusion_matrix(d["label"], d["pred"], labels=range(7))
mistakes = [(EMOTIONS[i], EMOTIONS[j], cm[i, j], cm[i, j] / cm[i].sum())
            for i in range(7) for j in range(7) if i != j]
mistakes = pd.DataFrame(mistakes, columns=["true", "predicted", "clips", "share of true class"])
print(mistakes.sort_values("clips", ascending=False).head(6).round(3).to_string(index=False))

# %% [markdown]
# ### 4.3 Normal vs strong emotional intensity
# RAVDESS actors speak every emotion at a normal and a strong intensity (4th number in the file name).
# Plain neutral has no strong version, so neutral clips are left out of this comparison.

# %%
def two_proportion_test(correct_a, correct_b):
    """Two-sided z-test: are two accuracies different?"""
    a, b = np.mean(correct_a), np.mean(correct_b)
    na, nb = len(correct_a), len(correct_b)
    pooled = (a * na + b * nb) / (na + nb)
    z = (a - b) / np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    return a, b, 2 * (1 - norm.cdf(abs(z)))


d = ens[(ens["experiment"] == "speaker") & (ens["emotion"] != "neutral")].copy()
d["intensity"] = d["path"].str.extract(r"03-01-\d\d-(\d\d)")[0].map({"01": "normal", "02": "strong"})
normal, strong, p = two_proportion_test(d[d["intensity"] == "normal"]["correct"],
                                        d[d["intensity"] == "strong"]["correct"])
print(f"Unseen speakers, ensemble: normal intensity {100 * normal:.1f}%, strong {100 * strong:.1f}% "
      f"(p = {p:.2g}, {(d['intensity'] == 'normal').sum()} vs {(d['intensity'] == 'strong').sum()} clips)")

# %% [markdown]
# ### 4.4 Per speaker (unseen speakers)

# %%
actors = (ens[ens["experiment"] == "speaker"].groupby(["speaker", "gender"])["correct"].mean()
          .reset_index().sort_values("correct"))
print(f"Per-actor accuracy: lowest {100 * actors['correct'].min():.1f}% "
      f"({', '.join(actors[actors['correct'] == actors['correct'].min()]['speaker'])}), highest "
      f"{100 * actors['correct'].max():.1f}% ({actors.iloc[-1]['speaker']}), median "
      f"{100 * actors['correct'].median():.1f}%")

# %%
fig, ax = plt.subplots(figsize=(8.5, 2.8))
colors = ["#2a78d6" if g == "male" else "#eb6834" for g in actors["gender"]]
ax.bar(range(len(actors)), 100 * actors["correct"], color=colors, width=0.7)
for g, c in [("male", "#2a78d6"), ("female", "#eb6834")]:
    mean = 100 * actors[actors["gender"] == g]["correct"].mean()
    ax.axhline(mean, color=c, ls="--", lw=1.2, label=f"{g} actors, mean {mean:.1f}%")
ax.set_xticks(range(len(actors)), actors["speaker"], fontsize=7, rotation=90)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#e6e5e0")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Ensemble accuracy for each of the 24 RAVDESS actors (never seen in training)", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "figures" / "fig3_actors.png", dpi=200)
plt.show()

# %% [markdown]
# ### 4.5 Unseen dataset (RAVDESS → TESS)

# %%
d = ens[ens["experiment"] == "ravdess_to_tess"]
share = d["pred"].map(dict(enumerate(EMOTIONS))).value_counts(normalize=True).reindex(EMOTIONS, fill_value=0)
print("Share of TESS clips predicted as each emotion (true share is 14.3% each):")
print((100 * share).round(1).to_string())
print("Accuracy per TESS speaker:", (100 * d.groupby("speaker")["correct"].mean()).round(1).to_dict())
offset = json.load(open(RESULTS / "tess_offset_check.json"))
for k, v in offset.items():
    print(f"{k}: padded frames {100 * v['padded_frames']:.1f}%, accuracy {100 * v['acc']:.1f}%, "
          f"predicted sad {100 * v['pred_sad_share']:.1f}%")

# %% [markdown]
# ### 4.6 Augment before split: why accuracy looks so high

# %%
d = ens[ens["experiment"] == "leaky"]
overlap = json.load(open(RESULTS / "leaky_overlap.json"))["test_samples_with_a_copy_in_train"]
print(f"Test samples with an augmented copy of the same recording in training: {100 * overlap:.1f}%")
print((100 * d.groupby("augmentation")["correct"].mean()).round(1).to_string())

# %% [markdown]
# ## 5. Bias check: male vs female speakers
# Unseen-speaker protocol: every clip is tested by a model that never heard that actor. RAVDESS is balanced
# (12 male and 12 female actors, 720 clips each), so a gap cannot come from unequal data.

# %%
rows = []
for model in MODELS[1:]:
    d = speaker[speaker["model"] == model]
    male, female = d[d["gender"] == "male"], d[d["gender"] == "female"]
    acc_m, acc_f, p = two_proportion_test(male["correct"], female["correct"])
    rows.append({"model": model, "male acc": 100 * acc_m, "female acc": 100 * acc_f,
                 "male macro-F1": 100 * scores(male)["macro_F1"], "female macro-F1": 100 * scores(female)["macro_F1"],
                 "gap (F - M)": 100 * (acc_f - acc_m), "p_value": p})
bias = pd.DataFrame(rows)
bias.to_csv(OUT / "gender_bias.csv", index=False)
print(bias.round(4).to_string(index=False))

# %%
male_acc = actors[actors["gender"] == "male"]["correct"]
female_acc = actors[actors["gender"] == "female"]["correct"]
test = mannwhitneyu(female_acc, male_acc, alternative="two-sided")
print(f"Actor level (ensemble): female mean {100 * female_acc.mean():.1f}%, male mean {100 * male_acc.mean():.1f}%, "
      f"Mann-Whitney U = {test.statistic:.0f}, p = {test.pvalue:.4f}")

d = speaker[speaker["model"] == "Ensemble"]
by_emotion = d.groupby(["emotion", "gender"])["correct"].mean().unstack().reindex(EMOTIONS)
by_emotion["gap (F - M)"] = by_emotion["female"] - by_emotion["male"]
print((100 * by_emotion).round(1))

# %%
d = main[(main["experiment"] == "paper") & (main["model"] == "Ensemble")]
acc_m, acc_f, p = two_proportion_test(d[d["gender"] == "male"]["correct"], d[d["gender"] == "female"]["correct"])
print(f"Random split (paper protocol), ensemble: male {100 * acc_m:.1f}%, female {100 * acc_f:.1f}%, p = {p:.2f} "
      f"({(d['gender'] == 'male').sum()} vs {(d['gender'] == 'female').sum()} clips)")

# %% [markdown]
# ## 6. Key results in one place

# %%
def acc(exp, model):
    """Accuracy (%) and 95% interval of one model under one protocol, as text."""
    r = metrics[(metrics["experiment"] == exp) & (metrics["model"] == model)].iloc[0]
    return f"{100 * r['accuracy']:.1f}% [{100 * r['ci_low']:.1f}–{100 * r['ci_high']:.1f}]"


for exp, name in PROTOCOLS.items():
    print(f"{name:22s} ensemble {acc(exp, 'Ensemble'):22s} SVM {acc(exp, 'SVM')}")
