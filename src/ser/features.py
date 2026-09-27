"""Hand-crafted acoustic features, following Chowdhury et al. (2025), Sec. "Data
augmentation and feature extraction process".

Paper settings reproduced here:
  * audio resampled to 22,050 Hz, 2.5 s loaded from an offset of 0.6 s
  * frame length 2,048, hop length 512 -> 108 frames
  * features: ZCR (1), RMSE (1), MFCC (20), Chroma STFT (12) per frame,
    flattened and concatenated -> 108 * 34 = 3,672 values per clip
  * augmentation: Gaussian noise (alpha = 0.035), pitch shift (0.7),
    noise + pitch, plus the original clip (4 versions)
  * missing values (clips shorter than 2.5 s) are filled with zeros

Features are stored as a (108, 34) frame matrix so that feature groups can be
removed for the ablation study; `flatten` produces the paper's 3,672-vector.
"""
import numpy as np
import librosa

SR = 22050
OFFSET = 0.6
DURATION = 2.5
N_FFT = 2048
HOP = 512
N_MFCC = 20
N_FRAMES = 1 + int(SR * DURATION) // HOP     # 108
NOISE_ALPHA = 0.035
PITCH_STEPS = 0.7

# Column ranges of each feature group inside the (T, 34) frame matrix.
FEATURE_GROUPS = {
    "zcr": slice(0, 1),
    "rms": slice(1, 2),
    "mfcc": slice(2, 22),
    "chroma": slice(22, 34),
}
N_FEATURES = 34

AUGMENTATIONS = ["original", "noise", "pitch", "noise_pitch"]


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True, offset=OFFSET, duration=DURATION)
    return y


def add_noise(y, rng):
    amp = NOISE_ALPHA * rng.uniform() * np.max(np.abs(y))
    return y + amp * rng.standard_normal(len(y))


def shift_pitch(y):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=PITCH_STEPS)


def augment(y, kind, rng):
    if kind == "original":
        return y
    if kind == "noise":
        return add_noise(y, rng)
    if kind == "pitch":
        return shift_pitch(y)
    if kind == "noise_pitch":
        return shift_pitch(add_noise(y, rng))
    raise ValueError(kind)


def frame_features(y):
    """Return a (N_FRAMES, 34) matrix: ZCR | RMSE | MFCC(20) | Chroma(12)."""
    y = y.astype(np.float32)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
    feats = np.vstack([zcr, rms, mfcc, chroma]).T
    out = np.zeros((N_FRAMES, N_FEATURES), dtype=np.float32)   # zero-fill short clips
    n = min(N_FRAMES, len(feats))
    out[:n] = feats[:n]
    return np.nan_to_num(out)


def extract_file(path, seed):
    """Return array (len(AUGMENTATIONS), N_FRAMES, 34) for one audio file."""
    rng = np.random.default_rng(seed)
    y = load_audio(path)
    return np.stack([frame_features(augment(y, kind, rng)) for kind in AUGMENTATIONS])


def flatten(x):
    """(N, T, F) -> (N, F*T) grouped by feature (all ZCR frames, then RMSE, ...),
    matching the paper's horizontally concatenated 3,672-value vector."""
    return x.transpose(0, 2, 1).reshape(len(x), -1)


def pooled(x):
    """(N, T, F) -> (N, 2F) mean and std over time, for the classical baselines."""
    return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1)
