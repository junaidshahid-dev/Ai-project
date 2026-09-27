"""Index both datasets and cache frame-level features (with augmentations).

Usage:
    python scripts/extract_features.py --ravdess data/RAVDESS --tess data/TESS
Output: features/{ravdess,tess}.npz and features/{ravdess,tess}.csv
"""
import argparse
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ser.data import index_ravdess, index_tess  # noqa: E402
from ser.features import extract_file  # noqa: E402


def _work(path):
    # Seed derived from the filename so augmentation is reproducible.
    return extract_file(path, seed=zlib.crc32(Path(path).name.encode()))


def run(df, name, out_dir, workers):
    t0 = time.time()
    with Pool(workers) as pool:
        feats = pool.map(_work, df["path"].tolist(), chunksize=8)
    X = np.stack(feats)  # (N, n_aug, T, F)
    np.savez_compressed(out_dir / f"{name}.npz", X=X)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} files -> {X.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess", default="data/RAVDESS")
    ap.add_argument("--tess", default="data/TESS")
    ap.add_argument("--out", default="features")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rav, tess = index_ravdess(a.ravdess), index_tess(a.tess)
    assert len(rav) == 1440, f"expected 1440 RAVDESS speech clips, got {len(rav)}"
    assert len(tess) == 2800, f"expected 2800 TESS clips, got {len(tess)}"
    run(rav, "ravdess", out, a.workers)
    run(tess, "tess", out, a.workers)
