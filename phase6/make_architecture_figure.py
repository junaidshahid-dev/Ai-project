"""Draw the 1D-CNN / CNN-BiLSTM architecture (paper Table 2) as phase6/figures/fig_architecture.png."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

LAYERS = [
    ("Input", "3,672 × 1", "#f0efec"),
    ("Conv1D\n128, k=5\nBatchNorm\nMaxPool /2", "1,836 × 128", "#cde2fb"),
    ("Conv1D\n128, k=5\nBatchNorm\nMaxPool /2\nDropout 0.2", "918 × 128", "#cde2fb"),
    ("Conv1D\n64, k=5\nBatchNorm\nMaxPool /2", "459 × 64", "#cde2fb"),
    ("Conv1D\n64, k=3\nBatchNorm\nMaxPool /2\nDropout 0.2", "230 × 64", "#cde2fb"),
    ("Bidirectional\nLSTM\n64 + 64 units\n(CNN-BiLSTM\nonly)", "230 × 128", "#fde2c8"),
    ("Conv1D\n32, k=3\nBatchNorm\nMaxPool /2\nDropout 0.2", "115 × 32", "#cde2fb"),
    ("Flatten", "3,680", "#f0efec"),
    ("Dense 128\nReLU\nBatchNorm", "128", "#d4f0e4"),
    ("Dense 7\nsoftmax", "7 emotions", "#d4f0e4"),
]

fig, ax = plt.subplots(figsize=(13, 3.2))
w, gap = 1.12, 0.22
for i, (name, shape, color) in enumerate(LAYERS):
    x = i * (w + gap)
    dashed = "LSTM" in name
    ax.add_patch(FancyBboxPatch((x, 0.9), w, 1.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="#52514e", linewidth=1, linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, 1.7, name, ha="center", va="center", fontsize=7.6, linespacing=1.25)
    ax.text(x + w / 2, 0.62, shape, ha="center", va="center", fontsize=7.6, color="#52514e")
    if i:
        ax.annotate("", xy=(x, 1.7), xytext=(x - gap, 1.7), arrowprops=dict(arrowstyle="->", color="#0b0b0b"))
ax.text(0, 2.95, "Output size of each layer is shown below the box. The 1D-CNN skips the orange BiLSTM layer "
        "(616,615 parameters); the CNN-BiLSTM keeps it (688,807 parameters).", fontsize=8.5)
ax.text(0, 0.12, "Ensemble = average of the 7 probabilities of the 1D-CNN and the CNN-BiLSTM; the predicted "
        "emotion is the one with the highest average probability.", fontsize=8.5)
ax.set_xlim(-0.1, len(LAYERS) * (w + gap))
ax.set_ylim(0, 3.2)
ax.axis("off")
plt.savefig(OUT / "fig_architecture.png", dpi=200, bbox_inches="tight")
print("saved", OUT / "fig_architecture.png")
