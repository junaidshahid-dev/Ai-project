"""Draw the pipeline diagram used in the Phase 3 report (phase3/figures/fig0_pipeline.png)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "fig0_pipeline.png"

fig, ax = plt.subplots(figsize=(12, 3.1))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.1); ax.axis("off")


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#333333", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))


DATA, PREP, MODEL, OUTC = "#e3eefa", "#fdf1d8", "#e2f3e4", "#f6e1e1"
top = 1.75
box(0.05, top, 1.45, 0.9, ".wav clip\n(RAVDESS / TESS)", DATA)
box(1.85, top, 1.65, 0.9, "2.5 s from 0.6 s\n22,050 Hz, mono", PREP)
box(3.85, top, 1.75, 0.9, "108 frames x 34\nZCR, RMS, MFCC,\nChroma (Phase 2)", PREP)
box(5.95, top, 1.75, 0.9, "split by clip;\naugment train x4\n(noise, pitch)", PREP)
box(8.05, top, 1.6, 0.9, "flatten 3,672\n+ z-score (train\nmean / std)", PREP)
box(10.0, 2.3, 0.95, 0.6, "CNN", MODEL)
box(10.0, 1.45, 0.95, 0.6, "CNN-\nBiLSTM", MODEL)
box(11.15, 1.75, 0.8, 0.9, "average\n-> emotion", OUTC)
for x1, x2 in [(1.5, 1.85), (3.5, 3.85), (5.6, 5.95), (7.7, 8.05)]:
    arrow(x1, top + 0.45, x2, top + 0.45)
arrow(9.65, top + 0.45, 10.0, 2.6); arrow(9.65, top + 0.45, 10.0, 1.75)
arrow(10.95, 2.6, 11.15, 2.35); arrow(10.95, 1.75, 11.15, 2.05)

box(5.95, 0.2, 1.75, 0.8, "mean + std over\ntime (68 values)", PREP)
box(8.05, 0.2, 1.9, 0.8, "SVM (RBF) /\nRandom Forest", MODEL)
box(10.25, 0.2, 1.7, 0.8, "baseline\nprediction", OUTC)
arrow(6.825, top, 6.825, 1.0); arrow(7.7, 0.6, 8.05, 0.6); arrow(9.95, 0.6, 10.25, 0.6)
ax.text(0.05, 0.55, "Top row: the paper's deep models\nBottom row: classical baselines",
        fontsize=8.2, va="center", color="#444444")

plt.tight_layout()
plt.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT)
