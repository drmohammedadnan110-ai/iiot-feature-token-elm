"""Figures computed from saved predictions and logs; no invented data."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import evaluate


def plot_predictions(predictions, classes, out, title="Executed evaluation"):
    frame = pd.read_csv(predictions) if not isinstance(predictions, pd.DataFrame) else predictions
    if frame.seed.nunique() != 1:
        raise ValueError("Plot one seed per figure")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = frame[[f"prob_{i}" for i in range(len(classes))]].to_numpy()
    result = evaluate(frame.y_true, p, classes)
    cm = np.asarray(result["confusion_matrix"])
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.62), max(6, len(classes) * 0.56)))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted class",
        ylabel="Reference class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = [b for b in result["bins"] if b["n"]]
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color="#0e7490")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Confidence",
        ylabel="Fraction correct",
        title=f"{title}\nECE = {result['ece']:.4f}",
    )
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, name in enumerate(classes):
        if 0 < (frame.y_true == i).sum() < len(frame):
            fpr, tpr, _ = roc_curve(frame.y_true == i, p[:, i])
            precision, recall, _ = precision_recall_curve(frame.y_true == i, p[:, i])
            axes[0].plot(fpr, tpr, label=name, lw=1)
            axes[1].plot(recall, precision, label=name, lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision–recall")
    axes[1].legend(fontsize=6, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "roc_pr.png", dpi=180)
    plt.close(fig)


def plot_history(path, out):
    frame = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame.epoch, frame.train_loss, "o-")
    axes[0].set(xlabel="Epoch", ylabel="TRAIN focal objective", ylim=(0, None))
    axes[1].plot(frame.epoch, frame.validation_macro_f1, "o-")
    axes[1].set(xlabel="Epoch", ylabel="VALIDATION macro-F1", ylim=(0, 1))
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
