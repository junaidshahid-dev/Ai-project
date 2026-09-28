"""Run the experiments and save per-clip predictions.

Experiments (each saves results/predictions_<exp>.csv and results/runs_<exp>.csv)
  paper    : replication of the paper's protocol on RAVDESS: random 80/10/10
             split of clips, augmentation applied to the training clips only
  leaky    : same, but augmentation is applied BEFORE the split, so augmented
             copies of a test clip can appear in training (tests hypothesis H2)
  speaker  : RAVDESS 4-fold speaker-independent CV (6 actors per fold,
             3 male + 3 female); every actor is tested exactly once (H3, H5)
  cross    : train on all RAVDESS -> test on TESS, and the reverse (H4)
  ablation : paper split, 1D-CNN and SVM, one feature group removed / MFCC
             only / no augmentation (H6)
  ablation_control : no augmentation with early-stopping patience 20

Usage:
    python scripts/run_experiments.py paper leaky speaker cross ablation ablation_control
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ser.data import EMOTIONS  # noqa: E402
from ser.features import FEATURE_GROUPS, N_FEATURES, flatten, pooled  # noqa: E402
from ser.models import baseline_models, ensemble_proba, predict_deep, train_deep  # noqa: E402

SEED = 42
N_CLASSES = len(EMOTIONS)
RESULTS = ROOT / "results"
FOLDS = [list(range(1, 7)), list(range(7, 13)), list(range(13, 19)), list(range(19, 25))]


def load(name):
    X = np.load(ROOT / "features" / f"{name}.npz")["X"]      # (N, 4, T, F)
    meta = pd.read_csv(ROOT / "features" / f"{name}.csv")
    return X, meta


def feature_columns(drop=None, only=None):
    cols = np.arange(N_FEATURES)
    if only:
        return cols[FEATURE_GROUPS[only]]
    if drop:
        return np.setdiff1d(cols, cols[FEATURE_GROUPS[drop]])
    return cols


def samples(X, idx, aug, cols):
    """Frame matrices for the given clips; aug=True adds the 3 augmented copies."""
    v = X[idx][:, :, :, cols] if aug else X[idx][:, :1, :, cols]
    k = v.shape[1]
    return v.reshape(len(idx) * k, *v.shape[2:]), k


def fit_predict(train, val, test, models, save_dir=None, tag="", patience=5):
    """train/val/test are (frames, labels) tuples; returns test probabilities and run info."""
    (F_tr, y_tr), (F_val, y_val), (F_te, _) = train, val, test
    probas, info = {}, {}

    if any(m in models for m in ("Majority", "SVM", "RandomForest")):
        P_tr, P_te = pooled(F_tr), pooled(F_te)
        for name, clf in baseline_models(SEED).items():
            if name not in models:
                continue
            t0 = time.time()
            clf.fit(P_tr, y_tr)
            fit_s = time.time() - t0
            t0 = time.time()
            p = np.zeros((len(F_te), N_CLASSES))
            p[:, clf.classes_] = clf.predict_proba(P_te)
            probas[name] = p
            info[name] = {"fit_s": fit_s, "infer_ms": 1000 * (time.time() - t0) / len(F_te)}

    # Paper pipeline: flattened 3,672-value vectors standardised per dimension
    # with statistics from the training set only.
    V_tr, V_val, V_te = flatten(F_tr), flatten(F_val), flatten(F_te)
    mu, sd = V_tr.mean(axis=0), V_tr.std(axis=0) + 1e-8
    V_tr, V_val, V_te = [(v - mu) / sd for v in (V_tr, V_val, V_te)]
    for name in ("CNN", "CNN_BiLSTM"):
        if name not in models:
            continue
        t0 = time.time()
        model, hist = train_deep(name, V_tr, y_tr, V_val, y_val, N_CLASSES, SEED, patience=patience)
        fit_s = time.time() - t0
        t0 = time.time()
        probas[name] = predict_deep(model, V_te)
        info[name] = {"fit_s": fit_s, "infer_ms": 1000 * (time.time() - t0) / len(V_te),
                      "params": int(model.count_params()), "epochs": len(hist["loss"]),
                      "best_val_acc": float(max(hist["val_accuracy"]))}
        if tag:
            with open(RESULTS / f"history_{tag}_{name}.json", "w") as f:
                json.dump({k: [float(v) for v in vals] for k, vals in hist.items()}, f)
        if save_dir:
            save_dir.mkdir(exist_ok=True)
            model.save(save_dir / f"{name}.keras")
        print(f"  {name}: {info[name]['epochs']} epochs, {fit_s / 60:.1f} min", flush=True)
    if "CNN" in probas and "CNN_BiLSTM" in probas:
        probas["Ensemble"] = ensemble_proba([probas["CNN"], probas["CNN_BiLSTM"]])
        info["Ensemble"] = {"params": info["CNN"]["params"] + info["CNN_BiLSTM"]["params"]}
    if save_dir:
        np.savez(save_dir / "normalizer.npz", mu=mu, sd=sd)
    return probas, info


def collect(out, exp, fold, meta, probas, info, variant="full", aug_kind=None):
    rows, runs = out
    for model, p in probas.items():
        df = meta[["path", "dataset", "speaker", "gender", "emotion", "label"]].copy().reset_index(drop=True)
        if aug_kind is not None:
            df["augmentation"] = aug_kind
        df["pred"] = p.argmax(axis=1)
        df["experiment"], df["fold"], df["model"], df["variant"] = exp, fold, model, variant
        rows.append(df)
        runs.append({"experiment": exp, "fold": fold, "model": model, "variant": variant, **info.get(model, {})})


def save(out, exp):
    rows, runs = out
    pd.concat(rows).to_csv(RESULTS / f"predictions_{exp}.csv", index=False)
    pd.DataFrame(runs).to_csv(RESULTS / f"runs_{exp}.csv", index=False)
    print(f"saved {exp}", flush=True)


def split_clips(y, idx):
    """Stratified 80/10/10 split of clip indices."""
    tr, rest = train_test_split(idx, test_size=0.2, stratify=y[idx], random_state=SEED)
    val, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=SEED)
    return tr, val, te


def val_holdout(y, tr):
    """Hold out 10% of the training clips (original audio only) for validation."""
    return train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)


ALL = ("Majority", "SVM", "RandomForest", "CNN", "CNN_BiLSTM")


def run(exp):
    Xr, mr = load("ravdess")
    yr = mr["label"].values
    cols = feature_columns()
    out = ([], [])

    if exp == "paper":
        tr, val, te = split_clips(yr, np.arange(len(mr)))
        F_tr, k = samples(Xr, tr, True, cols)
        p, i = fit_predict((F_tr, np.repeat(yr[tr], k)), (samples(Xr, val, False, cols)[0], yr[val]),
                           (samples(Xr, te, False, cols)[0], yr[te]), ALL, tag="paper")
        collect(out, exp, 0, mr.iloc[te], p, i)

    elif exp == "leaky":
        # Augment first, then split the pooled 4 x 1440 samples at random.
        F_all, k = samples(Xr, np.arange(len(mr)), True, cols)
        y_all = np.repeat(yr, k)
        clip = np.repeat(np.arange(len(mr)), k)
        kind = np.tile(["original", "noise", "pitch", "noise_pitch"], len(mr))
        tr, val, te = split_clips(y_all, np.arange(len(y_all)))
        p, i = fit_predict((F_all[tr], y_all[tr]), (F_all[val], y_all[val]), (F_all[te], y_all[te]),
                           ALL, tag="leaky")
        collect(out, exp, 0, mr.iloc[clip[te]], p, i, aug_kind=kind[te])
        overlap = np.isin(clip[te], clip[tr]).mean()
        with open(RESULTS / "leaky_overlap.json", "w") as f:
            json.dump({"test_samples_with_a_copy_in_train": float(overlap)}, f)

    elif exp == "speaker":
        for f, actors in enumerate(FOLDS):
            te = np.where(mr["actor"].isin(actors))[0]
            tr, val = val_holdout(yr, np.where(~mr["actor"].isin(actors))[0])
            F_tr, k = samples(Xr, tr, True, cols)
            p, i = fit_predict((F_tr, np.repeat(yr[tr], k)), (samples(Xr, val, False, cols)[0], yr[val]),
                               (samples(Xr, te, False, cols)[0], yr[te]), ALL, tag="speaker_f0" if f == 0 else "")
            collect(out, exp, f, mr.iloc[te], p, i)
            print(f"speaker fold {f} done", flush=True)

    elif exp == "cross":
        Xt, mt = load("tess")
        yt = mt["label"].values
        for name, (Xs, ys, ms), (Xd, yd, md), save_dir in [
            ("ravdess_to_tess", (Xr, yr, mr), (Xt, yt, mt), ROOT / "models"),
            ("tess_to_ravdess", (Xt, yt, mt), (Xr, yr, mr), None),
        ]:
            tr, val = val_holdout(ys, np.arange(len(ms)))
            F_tr, k = samples(Xs, tr, True, cols)
            te = np.arange(len(md))
            p, i = fit_predict((F_tr, np.repeat(ys[tr], k)), (samples(Xs, val, False, cols)[0], ys[val]),
                               (samples(Xd, te, False, cols)[0], yd), ALL, save_dir=save_dir)
            collect(out, name, 0, md, p, i)
            print(f"{name} done", flush=True)

    elif exp == "ablation":
        tr, val, te = split_clips(yr, np.arange(len(mr)))
        variants = [(f"no_{g}", feature_columns(drop=g), True) for g in FEATURE_GROUPS]
        variants += [("mfcc_only", feature_columns(only="mfcc"), True), ("no_augmentation", cols, False)]
        for name, c, aug in variants:
            F_tr, k = samples(Xr, tr, aug, c)
            p, i = fit_predict((F_tr, np.repeat(yr[tr], k)), (samples(Xr, val, False, c)[0], yr[val]),
                               (samples(Xr, te, False, c)[0], yr[te]), ("SVM", "CNN"))
            collect(out, "ablation", 0, mr.iloc[te], p, i, variant=name)
            print(f"ablation {name} done", flush=True)

    elif exp == "ablation_control":
        # Without augmentation an epoch has 4x fewer updates, so the paper's
        # early stopping (patience 5 epochs) can stop before the CNN has learned.
        # Control: no augmentation, patience 20 epochs (= 5 epochs of augmented data).
        tr, val, te = split_clips(yr, np.arange(len(mr)))
        p, i = fit_predict((samples(Xr, tr, False, cols)[0], yr[tr]), (samples(Xr, val, False, cols)[0], yr[val]),
                           (samples(Xr, te, False, cols)[0], yr[te]), ("CNN",), patience=20)
        collect(out, "ablation", 0, mr.iloc[te], p, i, variant="no_augmentation_patience20")

    save(out, exp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=["paper", "leaky", "speaker", "cross", "ablation", "ablation_control"])
    for e in ap.parse_args().experiments:
        RESULTS.mkdir(exist_ok=True)
        print(f"=== {e} ===", flush=True)
        run(e)
