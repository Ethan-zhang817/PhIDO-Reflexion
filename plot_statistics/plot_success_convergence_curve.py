"""Plot cumulative task success rate over retry rounds.

Use round 0 for the initial design attempt and rounds 1-3 for the three
allowed retries. Replace the example rates with measured cumulative pass rates.
The script saves both PNG and PDF outputs under ``plot_statistics/results``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "results"

ROUNDS = np.array([0, 1, 2, 3])
ROUND_LABELS = ["Initial", "Retry 1", "Retry 2", "Retry 3"]

# Cumulative final pass rate after allowing up to k retries.
REFLEXION_SUCCESS = np.array([46, 64, 75, 82], dtype=float)
RETRY_ONLY_SUCCESS = np.array([46, 54, 59, 62], dtype=float)

# Optional uncertainty, e.g. standard error or 95% CI half-width.
REFLEXION_ERROR = np.array([5, 5, 4, 4], dtype=float)
RETRY_ONLY_ERROR = np.array([5, 5, 5, 5], dtype=float)


def plot_success_convergence() -> None:
    """Render and save the convergence curve."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    ax.plot(
        ROUNDS,
        REFLEXION_SUCCESS,
        marker="o",
        markersize=7,
        linewidth=2.6,
        color="#2563eb",
        label="Reflexion",
    )
    ax.fill_between(
        ROUNDS,
        REFLEXION_SUCCESS - REFLEXION_ERROR,
        REFLEXION_SUCCESS + REFLEXION_ERROR,
        color="#2563eb",
        alpha=0.14,
        linewidth=0,
    )

    ax.plot(
        ROUNDS,
        RETRY_ONLY_SUCCESS,
        marker="s",
        markersize=6.5,
        linewidth=2.4,
        color="#f97316",
        label="Retry-only",
    )
    ax.fill_between(
        ROUNDS,
        RETRY_ONLY_SUCCESS - RETRY_ONLY_ERROR,
        RETRY_ONLY_SUCCESS + RETRY_ONLY_ERROR,
        color="#f97316",
        alpha=0.14,
        linewidth=0,
    )

    ax.annotate(
        "Max retry budget",
        xy=(3, max(REFLEXION_SUCCESS[-1], RETRY_ONLY_SUCCESS[-1])),
        xytext=(2.45, 94),
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.2},
        color="#475569",
        fontsize=10,
    )

    ax.set_xlabel("Number of retries allowed")
    ax.set_ylabel("Cumulative Success Rate (%)")
    ax.set_xticks(ROUNDS)
    ax.set_xticklabels(ROUND_LABELS)
    ax.set_xlim(-0.12, 3.12)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / "success_convergence_by_retry_round.png"
    pdf_path = OUTPUT_DIR / "success_convergence_by_retry_round.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    plot_success_convergence()
