"""Dataset indexing for RAVDESS and TESS.

Both datasets are mapped to a common 7-class label set. As in Chowdhury et al.
(2025, Table 1: 288 neutral RAVDESS clips), RAVDESS "calm" is merged into
"neutral", so all 1,440 RAVDESS speech clips are used.
"""
from pathlib import Path

import pandas as pd

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
LABEL_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}

# RAVDESS filename: modality-channel-emotion-intensity-statement-repetition-actor.wav
RAVDESS_CODES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

# TESS filename: {OAF|YAF}_{word}_{emotion}.wav
TESS_CODES = {
    "neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
    "fear": "fearful", "disgust": "disgust", "ps": "surprised",
}


def _unique_wavs(root):
    """Recursively list .wav files, de-duplicated by filename.

    The Kaggle copies of both datasets contain the same files twice in nested
    folders, so de-duplication is required to avoid train/test leakage.
    """
    seen = {}
    for p in sorted(Path(root).rglob("*.wav")):
        seen.setdefault(p.name.lower(), p)
    return list(seen.values())


def index_ravdess(root, merge_calm=True):
    rows = []
    for p in _unique_wavs(root):
        parts = p.stem.split("-")
        if len(parts) != 7 or parts[0] != "03":  # 03 = audio-only
            continue
        emotion = RAVDESS_CODES[parts[2]]
        if emotion == "calm":
            if not merge_calm:
                continue
            emotion = "neutral"
        actor = int(parts[6])
        rows.append({
            "path": str(p), "dataset": "RAVDESS", "emotion": emotion,
            "label": LABEL_TO_ID[emotion], "speaker": f"R{actor:02d}",
            "actor": actor, "gender": "male" if actor % 2 else "female",
            "intensity": "normal" if parts[3] == "01" else "strong",
        })
    return pd.DataFrame(rows)


def index_tess(root):
    rows = []
    for p in _unique_wavs(root):
        parts = p.stem.split("_")
        if len(parts) != 3 or parts[2].lower() not in TESS_CODES:
            continue
        emotion = TESS_CODES[parts[2].lower()]
        rows.append({
            "path": str(p), "dataset": "TESS", "emotion": emotion,
            "label": LABEL_TO_ID[emotion], "speaker": parts[0].upper(),
            "actor": -1, "gender": "female", "intensity": "n/a",
        })
    return pd.DataFrame(rows)
