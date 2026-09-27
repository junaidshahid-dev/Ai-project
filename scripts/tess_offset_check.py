"""Controlled check of the zero-padding explanation for the RAVDESS -> TESS collapse.

The saved RAVDESS-trained models (models/, written by the cross experiment) are
evaluated on TESS features extracted with offset 0.6 s (paper setting, ~42% of
frames zero-padded) and with offset 0.0 s (less padding). Only the test-side
input changes; the models are identical.

Usage: python scripts/tess_offset_check.py
Output: results/tess_offset_check.json
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ser import features as F  # noqa: E402
from ser.data import EMOTIONS  # noqa: E402


def extract(args):
    path, offset = args
    y, _ = librosa.load(path, sr=F.SR, mono=True, offset=offset, duration=F.DURATION)
    return F.frame_features(y)


def main():
    meta = pd.read_csv(ROOT / "features" / "tess.csv")
    norm = np.load(ROOT / "models" / "normalizer.npz")
    nets = [tf.keras.models.load_model(ROOT / "models" / f"{n}.keras") for n in ("CNN", "CNN_BiLSTM")]
    out = {}
    for offset in (0.6, 0.0):
        with Pool(4) as pool:
            X = np.stack(pool.map(extract, [(p, offset) for p in meta["path"]], chunksize=16))
        pad = float((np.abs(X).sum(-1) == 0).mean())
        V = (F.flatten(X) - norm["mu"]) / norm["sd"]
        proba = np.mean([m.predict(V[..., None], batch_size=256, verbose=0) for m in nets], axis=0)
        pred = proba.argmax(1)
        out[f"offset_{offset}"] = {
            "padded_frames": pad,
            "acc": accuracy_score(meta["label"], pred),
            "f1_macro": f1_score(meta["label"], pred, average="macro"),
            "pred_sad_share": float((pred == EMOTIONS.index("sad")).mean()),
        }
        print(offset, out[f"offset_{offset}"], flush=True)
    with open(ROOT / "results" / "tess_offset_check.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
