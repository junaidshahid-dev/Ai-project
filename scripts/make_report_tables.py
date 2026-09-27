"""Generate LaTeX macros and tables for the report from results/ (run after analyze.py).

Every number in the report comes from here, so the report cannot drift from the results.
Output: report/generated/numbers.tex, report/generated/tab_*.tex
"""
import json

import numpy as np
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "report" / "generated"
LABELS = {"Majority": "Majority class", "SVM": "SVM (RBF)", "RandomForest": "Random Forest",
          "CNN": "1D-CNN", "CNN_BiLSTM": "CNN\\_Bi-LSTM", "Ensemble": "Averaging ensemble"}
MODELS = list(LABELS)
MACRO_NAMES = {"Majority": "Maj", "SVM": "Svm", "RandomForest": "Rf", "CNN": "Cnn",
               "CNN_BiLSTM": "Lstm", "Ensemble": "Ens"}
EXP_NAMES = {"paper": "Paper", "leaky": "Leaky", "speaker": "Spk",
             "ravdess_to_tess": "RT", "tess_to_ravdess": "TR"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.load(open(ROOT / "results" / "summary.json"))
    comp = pd.read_csv(TAB / "model_comparison.csv")
    macros = []

    def mac(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # Accuracy / F1 macros for every experiment x model, e.g. \AccSpkEns
    for _, r in comp.iterrows():
        k = EXP_NAMES[r["experiment"]] + MACRO_NAMES[r["model"]]
        mac(f"Acc{k}", pct(r["acc"]))
        mac(f"Fone{k}", pct(r["f1_macro"]))
        mac(f"Fw{k}", pct(r["f1_weighted"]))
        if r["experiment"] == "speaker":
            mac(f"AccStd{k}", pct(r["acc_fold_std"]))

    # Table: main comparison (accuracy / macro-F1 in %)
    cols = [e for e in EXP_NAMES if e in set(comp["experiment"])]
    head = {"paper": "Paper protocol", "leaky": "Aug.-before-split", "speaker": "Speaker-indep.",
            "ravdess_to_tess": "RAV$\\to$TESS", "tess_to_ravdess": "TESS$\\to$RAV"}
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Model & " + " & ".join(head[c] for c in cols) + " \\\\", "\\midrule"]
    best = {c: comp[(comp["experiment"] == c) & (comp["model"] != "Majority")]["acc"].max() for c in cols}
    for m in MODELS:
        cells = []
        for c in cols:
            r = comp[(comp["experiment"] == c) & (comp["model"] == m)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            acc = pct(r["acc"])
            if c == "speaker":
                acc += f"\\,$\\pm$\\,{pct(r['acc_fold_std'])}"
            cell = f"{acc} / {pct(r['f1_macro'])}"
            majority = comp[(comp["experiment"] == c) & (comp["model"] == "Majority")]["acc"].iloc[0]
            if abs(r["acc"] - best[c]) < 1e-12 and m != "Majority" and r["acc"] > majority:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        if m == "CNN":
            lines.append("\\midrule")
        lines.append(f"{LABELS[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")

    # Table: replication vs paper
    rep = pd.DataFrame(s["replication"])
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Reported in paper} & \\multicolumn{2}{c}{Ours: paper protocol}"
             " & \\multicolumn{2}{c}{Ours: aug.-before-split} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Model & Acc. & W-F1 & Acc. & W-F1 & Acc. & W-F1 \\\\", "\\midrule"]
    for _, r in rep.iterrows():
        def g(k):
            return f"{r[k]:.2f}" if k in r and pd.notna(r[k]) else "--"
        lines.append(f"{LABELS[r['model']]} & {g('reported_acc')} & {g('reported_f1w')} & "
                     f"{g('ours_paper_acc')} & {g('ours_paper_f1w')} & {g('ours_leaky_acc')} & {g('ours_leaky_f1w')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_replication.tex").write_text("\n".join(lines) + "\n")

    # Leakage details
    if "leaky_by_augmentation" in s:
        for k, v in s["leaky_by_augmentation"].items():
            mac("LeakyAcc" + k.replace("_", "").capitalize(), pct(v))
        mac("LeakyOverlap", pct(s["leaky_overlap"]["test_samples_with_a_copy_in_train"]))

    # Gender bias
    if "gender_bias" in s:
        b = pd.DataFrame(s["gender_bias"])
        lines = ["\\begin{tabular}{lccccc}", "\\toprule",
                 "Model & Acc. male & Acc. female & Gap (pp) & $z$ & $p$ \\\\", "\\midrule"]
        for _, r in b.iterrows():
            p = f"{r['p_value']:.3f}" if r["p_value"] >= 0.001 else "$<$0.001"
            lines.append(f"{LABELS[r['model']]} & {pct(r['acc_male'])} & {pct(r['acc_female'])} & "
                         f"{r['gap_pp']:+.1f} & {r['z']:.2f} & {p} \\\\")
            k = MACRO_NAMES[r["model"]]
            mac(f"GenderMale{k}", pct(r["acc_male"]))
            mac(f"GenderFemale{k}", pct(r["acc_female"]))
            mac(f"GenderGap{k}", f"{r['gap_pp']:+.1f}")
            mac(f"GenderP{k}", p)
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / "tab_gender.tex").write_text("\n".join(lines) + "\n")

        pe = pd.DataFrame(s["gender_per_emotion"])
        lines = ["\\begin{tabular}{lccc}", "\\toprule",
                 "Emotion & Recall male & Recall female & Gap (pp) \\\\", "\\midrule"]
        for _, r in pe.iterrows():
            lines.append(f"{r['emotion'].capitalize()} & {pct(r['recall_male'])} & {pct(r['recall_female'])} & "
                         f"{100 * (r['recall_female'] - r['recall_male']):+.1f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / "tab_gender_emotion.tex").write_text("\n".join(lines) + "\n")
        acts = s["speaker_per_actor_acc"]
        mac("ActorAccMin", pct(min(acts.values())))
        mac("ActorAccMax", pct(max(acts.values())))
        mac("ActorMin", min(acts, key=acts.get).replace("R", "Actor~"))
        mac("ActorMax", max(acts, key=acts.get).replace("R", "Actor~"))

    if "gender_actor_level" in s:
        ga = s["gender_actor_level"]
        mac("GenderActorU", f"{ga['U']:.0f}")
        mac("GenderActorP", f"{ga['p_value']:.3f}")
        c = s["speaker_top_confusions"]
        for i, (a, b, _, share) in enumerate(c[:3]):
            name = ["One", "Two", "Three"][i]
            mac(f"Conf{name}", f"{a}$\\to${b}")
            mac(f"Conf{name}Pct", pct(share))
        rec = s["speaker_recall"]
        mac("SpkRecallMin", pct(min(rec.values())))
        mac("SpkRecallMinName", min(rec, key=rec.get))
        mac("SpkRecallMax", pct(max(rec.values())))
        mac("SpkRecallMaxName", max(rec, key=rec.get))

    # Cross-dataset details
    if "tess_per_speaker" in s:
        for spk, v in s["tess_per_speaker"].items():
            mac(f"TessAcc{spk.capitalize()}", pct(v))
        dist = s["tess_pred_distribution"]
        top = max(dist, key=dist.get)
        mac("TessTopPred", top)
        mac("TessTopPredShare", pct(dist[top]))
        rec = s["tess_per_emotion_recall"]
        lines = ["\\begin{tabular}{" + "c" * len(rec) + "}", "\\toprule",
                 " & ".join(e[:3].capitalize() + "." for e in rec) + " \\\\", "\\midrule",
                 " & ".join(pct(v) for v in rec.values()) + " \\\\", "\\bottomrule", "\\end{tabular}"]
        (OUT / "tab_tess_recall.tex").write_text("\n".join(lines) + "\n")

    if "tess_offset_check" in s:
        oc = s["tess_offset_check"]
        mac("OffsetAccPaper", pct(oc["offset_0.6"]["acc"]))
        mac("OffsetAccZero", pct(oc["offset_0.0"]["acc"]))
        mac("OffsetFoneZero", pct(oc["offset_0.0"]["f1_macro"]))
        mac("OffsetPadPaper", pct(oc["offset_0.6"]["padded_frames"], 0))
        mac("OffsetPadZero", pct(oc["offset_0.0"]["padded_frames"], 0))
        mac("OffsetSadPaper", pct(oc["offset_0.6"]["pred_sad_share"], 0))
        mac("OffsetSadZero", pct(oc["offset_0.0"]["pred_sad_share"], 0))
    if "rav_pred_distribution" in s:
        dist = s["rav_pred_distribution"]
        top = max(dist, key=dist.get)
        mac("RavTopPred", top)
        mac("RavTopPredShare", pct(dist[top]))

    # Ablation
    if "ablation" in s:
        a = pd.DataFrame(s["ablation"])
        names = {"full": "All features (ZCR+RMSE+MFCC+Chroma)", "no_zcr": "without ZCR",
                 "no_rms": "without RMSE", "no_mfcc": "without MFCC", "no_chroma": "without Chroma",
                 "mfcc_only": "MFCC only", "no_augmentation": "All features, no augmentation"}
        dims = {"full": 3672, "no_zcr": 3564, "no_rms": 3564, "no_mfcc": 1512, "no_chroma": 2376,
                "mfcc_only": 2160, "no_augmentation": 3672}
        lines = ["\\begin{tabular}{lcccc}", "\\toprule",
                 "Variant & Input dim. & SVM acc. & 1D-CNN acc. & $\\Delta$ CNN (pp) \\\\", "\\midrule"]
        for v in names:
            r = a[a["variant"] == v]
            if r.empty:
                continue
            sv = r[r["model"] == "SVM"]
            cn = r[r["model"] == "CNN"]
            svs = pct(sv["acc"].iloc[0]) if len(sv) else "--"
            cns = pct(cn["acc"].iloc[0]) if len(cn) else "--"
            d = f"{cn['delta_pp'].iloc[0]:+.1f}" if len(cn) and v != "full" else "--"
            lines.append(f"{names[v]} & {dims[v]:,} & {svs} & {cns} & {d} \\\\".replace(",", "{,}"))
            if len(cn) and v != "full":
                mac("Abl" + v.replace("_", "").capitalize(), f"{cn['delta_pp'].iloc[0]:+.1f}")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (OUT / "tab_ablation.tex").write_text("\n".join(lines) + "\n")

    # Efficiency
    eff = pd.DataFrame(s["efficiency"])
    for _, r in eff.iterrows():
        k = MACRO_NAMES[r["model"]]
        if pd.notna(r.get("params")):
            mac(f"Params{k}", f"{int(r['params']):,}".replace(",", "{,}"))
        if pd.notna(r.get("fit_s")):
            mac(f"FitMin{k}", f"{r['fit_s'] / 60:.1f}")
        if pd.notna(r.get("infer_ms")):
            mac(f"InferMs{k}", f"{r['infer_ms']:.2f}")
        if pd.notna(r.get("epochs")):
            mac(f"Epochs{k}", f"{int(r['epochs'])}")

    # Differences quoted in the text (computed, never typed by hand)
    def acc(exp, model):
        return comp[(comp["experiment"] == exp) & (comp["model"] == model)]["acc"].iloc[0]
    mac("LeakGainEns", f"{100 * (acc('leaky', 'Ensemble') - acc('paper', 'Ensemble')):.1f}")
    mac("LeakGainSvm", f"{100 * (acc('leaky', 'SVM') - acc('paper', 'SVM')):.1f}")
    mac("ReportedGapEns", f"{97.57 - 100 * acc('paper', 'Ensemble'):.1f}")
    if "speaker" in set(comp["experiment"]):
        mac("SpkDropEns", f"{100 * (acc('paper', 'Ensemble') - acc('speaker', 'Ensemble')):.1f}")
        mac("SpkEnsOverSvm", f"{100 * (acc('speaker', 'Ensemble') - acc('speaker', 'SVM')):.1f}")
    mac("PaperEnsOverSvm", f"{100 * (acc('paper', 'Ensemble') - acc('paper', 'SVM')):.1f}")
    if "tess_offset_check" in s:
        oc = s["tess_offset_check"]
        mac("OffsetGain", f"{100 * (oc['offset_0.0']['acc'] - oc['offset_0.6']['acc']):.1f}")

    # Compute budget and input statistics quoted in the text
    runs = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "results").glob("runs_*.csv"))])
    deep = runs[runs["model"].isin(["CNN", "CNN_BiLSTM"])]
    mac("NumTrainings", str(len(deep)))
    mac("ComputeHours", f"{deep['fit_s'].sum() / 3600:.0f}")

    X = np.load(ROOT / "features" / "tess.npz")["X"][:, 0]
    mac("TessPadPct", f"{100 * (np.abs(X).sum(-1) == 0).mean():.0f}")
    X = np.load(ROOT / "features" / "ravdess.npz")["X"][:, 0]
    mac("RavPadPct", f"{100 * (np.abs(X).sum(-1) == 0).mean():.0f}")

    (OUT / "numbers.tex").write_text("\n".join(macros) + "\n")
    print(f"wrote {len(macros)} macros and {len(list(OUT.glob('tab_*.tex')))} tables to {OUT}")


if __name__ == "__main__":
    main()
