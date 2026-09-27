"""Compute metrics, the gender-bias check and all figures from saved predictions.

Usage:
    python scripts/analyze.py
Output: results/tables/*.csv, results/summary.json, results/figures/*.pdf|png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, norm  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ser.data import EMOTIONS  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
MODELS = ["Majority", "SVM", "RandomForest", "CNN", "CNN_BiLSTM", "Ensemble"]
LABELS = {"Majority": "Majority", "SVM": "SVM", "RandomForest": "Random Forest",
          "CNN": "1D-CNN", "CNN_BiLSTM": "CNN-BiLSTM", "Ensemble": "Ensemble"}

# Reference palette (dataviz skill): blue / orange for categorical, gray for context.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def two_prop_z(a1, n1, a2, n2):
    p = (a1 * n1 + a2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (a1 - a2) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# Reported in Chowdhury et al. (2025), Table 4 (RAVDESS and TESS rows).
PAPER = {
    ("RAVDESS", "CNN"): (96.18, 96.22), ("RAVDESS", "CNN_BiLSTM"): (97.57, 97.29),
    ("RAVDESS", "Ensemble"): (97.57, 97.56), ("TESS", "CNN"): (100.00, 100.00),
    ("TESS", "CNN_BiLSTM"): (99.82, 100.00), ("TESS", "Ensemble"): (100.00, 100.00),
}
EXPERIMENTS = [("paper", "Paper protocol"), ("leaky", "Augment-before-split"),
               ("speaker", "Speaker-independent"), ("ravdess_to_tess", "RAVDESS \u2192 TESS"),
               ("tess_to_ravdess", "TESS \u2192 RAVDESS")]


def metrics(df):
    kw = dict(labels=range(len(EMOTIONS)), zero_division=0)
    return {"acc": accuracy_score(df["label"], df["pred"]),
            "f1_macro": f1_score(df["label"], df["pred"], average="macro", **kw),
            "f1_weighted": f1_score(df["label"], df["pred"], average="weighted", **kw),
            "n": len(df)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    preds = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("predictions_*.csv"))], ignore_index=True)
    runs = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("runs_*.csv"))], ignore_index=True)
    main_p = preds[preds["variant"] == "full"]
    have = [e for e, _ in EXPERIMENTS if e in set(main_p["experiment"])]
    summary = {}

    # ---- Model comparison across protocols ----------------------------------
    rows = []
    for exp in have:
        for m in MODELS:
            d = main_p[(main_p["experiment"] == exp) & (main_p["model"] == m)]
            if d.empty:
                continue
            r = {"experiment": exp, "model": m, **metrics(d)}
            if exp == "speaker":  # mean and std over the 4 speaker folds
                fa = [accuracy_score(g["label"], g["pred"]) for _, g in d.groupby("fold")]
                r.update(acc_fold_mean=np.mean(fa), acc_fold_std=np.std(fa))
            rows.append(r)
    comp = pd.DataFrame(rows)
    comp.to_csv(TAB / "model_comparison.csv", index=False)
    summary["model_comparison"] = comp.to_dict(orient="records")

    # ---- Replication vs paper -------------------------------------------------
    rep = []
    for m in ("CNN", "CNN_BiLSTM", "Ensemble"):
        r = {"model": m, "reported_acc": PAPER[("RAVDESS", m)][0], "reported_f1w": PAPER[("RAVDESS", m)][1]}
        for exp in ("paper", "leaky"):
            c = comp[(comp["experiment"] == exp) & (comp["model"] == m)]
            if len(c):
                r[f"ours_{exp}_acc"] = 100 * c["acc"].iloc[0]
                r[f"ours_{exp}_f1w"] = 100 * c["f1_weighted"].iloc[0]
        rep.append(r)
    pd.DataFrame(rep).to_csv(TAB / "replication.csv", index=False)
    summary["replication"] = rep

    if "leaky" in have:
        lk = main_p[(main_p["experiment"] == "leaky") & (main_p["model"] == "Ensemble")]
        summary["leaky_by_augmentation"] = {k: float((g["pred"] == g["label"]).mean())
                                            for k, g in lk.groupby("augmentation")}
        summary["leaky_overlap"] = json.load(open(RES / "leaky_overlap.json"))

    # ---- Efficiency -----------------------------------------------------------
    eff = runs[(runs["variant"] == "full") & (runs["experiment"] == "paper")].set_index("model")
    eff = eff.reindex([m for m in MODELS if m in eff.index])[["params", "fit_s", "infer_ms", "epochs"]]
    eff.to_csv(TAB / "efficiency.csv")
    summary["efficiency"] = eff.reset_index().to_dict(orient="records")

    # ---- Bias check: male vs female (speaker-independent, all 24 actors) ----
    bias = pd.DataFrame()
    if "speaker" in have:
        rows = []
        e2 = main_p[main_p["experiment"] == "speaker"]
        for m in MODELS[1:]:
            d = e2[e2["model"] == m]
            gm, gf = metrics(d[d["gender"] == "male"]), metrics(d[d["gender"] == "female"])
            z, pval = two_prop_z(gm["acc"], gm["n"], gf["acc"], gf["n"])
            rows.append({"model": m, "acc_male": gm["acc"], "acc_female": gf["acc"],
                         "f1_male": gm["f1_macro"], "f1_female": gf["f1_macro"],
                         "gap_pp": 100 * (gf["acc"] - gm["acc"]), "z": z, "p_value": pval})
        bias = pd.DataFrame(rows)
        bias.to_csv(TAB / "gender_bias.csv", index=False)
        summary["gender_bias"] = bias.to_dict(orient="records")
        d = e2[e2["model"] == "Ensemble"]
        per_emo = []
        for e in EMOTIONS:
            row = {"emotion": e}
            for g in ("male", "female"):
                s_ = d[(d["emotion"] == e) & (d["gender"] == g)]
                row[f"recall_{g}"] = float((s_["pred"] == s_["label"]).mean())
            per_emo.append(row)
        pd.DataFrame(per_emo).to_csv(TAB / "gender_per_emotion.csv", index=False)
        summary["gender_per_emotion"] = per_emo
        summary["speaker_per_actor_acc"] = {s_: float((g["pred"] == g["label"]).mean())
                                            for s_, g in d.groupby("speaker")}
        # Actor-level test: clips of one actor are not independent, so also compare
        # the 12 male vs 12 female per-actor accuracies (Mann-Whitney U, two-sided).
        acts = d.groupby(["speaker", "gender"]).apply(lambda g: (g["pred"] == g["label"]).mean()).reset_index(name="acc")
        u = mannwhitneyu(acts[acts["gender"] == "female"]["acc"], acts[acts["gender"] == "male"]["acc"],
                         alternative="two-sided")
        summary["gender_actor_level"] = {
            "male_mean": float(acts[acts["gender"] == "male"]["acc"].mean()),
            "female_mean": float(acts[acts["gender"] == "female"]["acc"].mean()),
            "U": float(u.statistic), "p_value": float(u.pvalue)}
        # Most frequent confusions (speaker-independent ensemble)
        cm = confusion_matrix(d["label"], d["pred"], labels=range(len(EMOTIONS)))
        conf = [(EMOTIONS[i], EMOTIONS[j], int(cm[i, j]), float(cm[i, j] / cm[i].sum()))
                for i in range(len(EMOTIONS)) for j in range(len(EMOTIONS)) if i != j]
        summary["speaker_top_confusions"] = sorted(conf, key=lambda t: -t[3])[:5]
        summary["speaker_recall"] = {EMOTIONS[i]: float(cm[i, i] / cm[i].sum()) for i in range(len(EMOTIONS))}

    if "ravdess_to_tess" in have:
        x = main_p[(main_p["experiment"] == "ravdess_to_tess") & (main_p["model"] == "Ensemble")]
        summary["tess_per_speaker"] = {s_: float((g["pred"] == g["label"]).mean()) for s_, g in x.groupby("speaker")}
        summary["tess_pred_distribution"] = x["pred"].map(dict(enumerate(EMOTIONS))).value_counts(normalize=True).to_dict()
        summary["tess_per_emotion_recall"] = {e: float((g["pred"] == g["label"]).mean()) for e, g in x.groupby("emotion")}
        if (RES / "tess_offset_check.json").exists():  # controlled padding test
            summary["tess_offset_check"] = json.load(open(RES / "tess_offset_check.json"))
    if "tess_to_ravdess" in have:
        x = main_p[(main_p["experiment"] == "tess_to_ravdess") & (main_p["model"] == "Ensemble")]
        summary["rav_pred_distribution"] = x["pred"].map(dict(enumerate(EMOTIONS))).value_counts(normalize=True).to_dict()

    # ---- Ablation (paper split) ---------------------------------------------
    abl = pd.DataFrame()
    if (preds["experiment"] == "ablation").any():
        rows = []
        base = main_p[(main_p["experiment"] == "paper") & main_p["model"].isin(["SVM", "CNN"])]
        for m, g in base.groupby("model"):
            rows.append({"variant": "full", "model": m, **metrics(g)})
        for (v, m), g in preds[preds["experiment"] == "ablation"].groupby(["variant", "model"]):
            rows.append({"variant": v, "model": m, **metrics(g)})
        abl = pd.DataFrame(rows)
        ref = abl[abl["variant"] == "full"].set_index("model")["acc"]
        abl["delta_pp"] = 100 * (abl["acc"] - abl["model"].map(ref))
        abl.to_csv(TAB / "ablation.csv", index=False)
        summary["ablation"] = abl.to_dict(orient="records")

    with open(RES / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- Figures -------------------------------------------------------------
    panels = [(e, t) for e, t in EXPERIMENTS if e in have and e != "tess_to_ravdess"]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 1.2, 2.5), sharey=True, squeeze=False)
    for ax, (exp, title) in zip(axes[0], panels):
        c = comp[comp["experiment"] == exp].set_index("model").reindex(MODELS)
        colors = [GRAY if m in ("Majority", "SVM", "RandomForest") else BLUE for m in MODELS]
        ax.barh(range(len(MODELS)), 100 * c["acc"], color=colors, height=0.6)
        for i, v in enumerate(100 * c["acc"]):
            ax.text(v + 2, i, f"{v:.1f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 50, 100])
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("Accuracy (%)")
        ax.grid(axis="y", visible=False)
    axes[0][0].invert_yaxis()  # shared y-axis: invert once, not per panel
    save(fig, "fig_model_comparison")

    cm_panels = [(e, t) for e, t in EXPERIMENTS if e in have and e in ("paper", "speaker", "ravdess_to_tess")]
    fig, axes = plt.subplots(1, len(cm_panels), figsize=(2.55 * len(cm_panels), 2.9), squeeze=False)
    short = ["neu", "hap", "sad", "ang", "fea", "dis", "sur"]
    for ax, (exp, title) in zip(axes[0], cm_panels):
        d = main_p[(main_p["experiment"] == exp) & (main_p["model"] == "Ensemble")]
        cm = confusion_matrix(d["label"], d["pred"], labels=range(len(EMOTIONS)), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(EMOTIONS)), short, fontsize=7)
        ax.set_yticks(range(len(EMOTIONS)), short, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_title(title, fontsize=8.5)
        ax.grid(False)
    axes[0][0].set_ylabel("True")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_confusion")

    if len(bias):
        fig, ax = plt.subplots(figsize=(4.6, 2.5))
        y = np.arange(len(bias))
        ax.barh(y - 0.18, 100 * bias["acc_male"], height=0.34, color=BLUE, label="Male actors")
        ax.barh(y + 0.18, 100 * bias["acc_female"], height=0.34, color=ORANGE, label="Female actors")
        for i, r in bias.iterrows():
            ax.text(100 * r["acc_male"] + 1, i - 0.18, f"{100 * r['acc_male']:.1f}", va="center", fontsize=7)
            ax.text(100 * r["acc_female"] + 1, i + 0.18, f"{100 * r['acc_female']:.1f}", va="center", fontsize=7)
        ax.set_yticks(y, [LABELS[m] for m in bias["model"]])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy (%), speaker-independent CV")
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
        ax.grid(axis="y", visible=False)
        save(fig, "fig_gender_bias")

    if len(abl):
        names = {"no_zcr": "without ZCR", "no_rms": "without RMSE", "no_chroma": "without Chroma",
                 "no_mfcc": "without MFCC", "mfcc_only": "MFCC only", "no_augmentation": "without augmentation"}
        a = abl[(abl["model"] == "CNN") & (abl["variant"] != "full")].sort_values("delta_pp")
        fig, ax = plt.subplots(figsize=(4.6, 2.3))
        ax.barh(range(len(a)), a["delta_pp"], color=BLUE, height=0.55)
        for i, v in enumerate(a["delta_pp"]):
            ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(a)), [names[v] for v in a["variant"]])
        ax.set_xlim(min(a["delta_pp"].min() * 1.35, -2), max(a["delta_pp"].max() * 1.35, 2))
        ax.set_xlabel("Change in 1D-CNN accuracy vs. full model (pp)")
        ax.grid(axis="y", visible=False)
        save(fig, "fig_ablation")

    hist_files = [RES / f"history_paper_{m}.json" for m in ("CNN", "CNN_BiLSTM")]
    if all(h.exists() for h in hist_files):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
        for ax, m, hf in zip(axes, ("CNN", "CNN_BiLSTM"), hist_files):
            h = json.load(open(hf))
            ax.plot(h["loss"], color=BLUE, linewidth=2, label="train")
            ax.plot(h["val_loss"], color=ORANGE, linewidth=2, label="validation")
            ax.set_title(LABELS[m], fontsize=9)
            ax.set_xlabel("Epoch")
            ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("Cross-entropy loss")
        save(fig, "fig_training_curves")

    pd.set_option("display.width", 200)
    print(comp.round(4).to_string())
    print(pd.DataFrame(rep).round(2).to_string())
    if len(bias):
        print(bias.round(4).to_string())
    if len(abl):
        print(abl.round(4).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("tess", "leaky", "speaker_per"))}, indent=1, default=float))


if __name__ == "__main__":
    main()
