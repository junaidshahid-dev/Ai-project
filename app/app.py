"""Streamlit demo (bonus): upload a .wav clip and get the predicted emotion.

Run after `python scripts/run_experiments.py cross` has written models/:
    streamlit run app/app.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ser.data import EMOTIONS  # noqa: E402
from ser.features import flatten, frame_features, load_audio  # noqa: E402

MODEL_DIR = ROOT / "models"


@st.cache_resource
def load_models():
    norm = np.load(MODEL_DIR / "normalizer.npz")
    nets = [tf.keras.models.load_model(MODEL_DIR / f"{n}.keras") for n in ("CNN", "CNN_BiLSTM")]
    return nets, norm["mu"], norm["sd"]


st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("Speech Emotion Recognition")
st.caption("Lightweight CNN + CNN-BiLSTM ensemble on hand-crafted features (ZCR, RMS, Chroma, MFCC), "
           "trained on RAVDESS (pipeline of Chowdhury et al., 2025). F26-11, AI2002.")

if not (MODEL_DIR / "normalizer.npz").exists():
    st.error("No trained models found. Run `python scripts/run_experiments.py cross` first.")
    st.stop()

nets, mu, sd = load_models()
upload = st.file_uploader("Upload a speech clip (.wav)", type=["wav"])
if upload:
    st.audio(upload)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(upload.getvalue())
        tmp.flush()
        x = flatten(frame_features(load_audio(tmp.name))[None])
    x = ((x - mu) / sd)[..., None]
    proba = np.mean([m.predict(x, verbose=0)[0] for m in nets], axis=0)
    top = int(proba.argmax())
    st.subheader(f"Predicted emotion: **{EMOTIONS[top]}** ({100 * proba[top]:.1f}%)")
    st.bar_chart(pd.DataFrame({"probability": proba}, index=EMOTIONS))
    st.info("The model was trained on acted North-American English speech; accuracy on other "
            "speakers, languages or recording conditions is substantially lower (see report, Section 4).")
