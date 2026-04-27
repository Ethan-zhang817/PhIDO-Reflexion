"""Plot final pass rate by model configuration and task difficulty.

Replace ``PASS_RATES`` and ``ERROR_BARS`` with measured experiment results.
The script saves both PNG and PDF outputs under ``plot_statistics/results``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "results"

CONFIGS = [
    "GPT-4.1-mini\n(Base)",
    "GPT-4.1-mini\n(Reflexion)",
    "GPT-4o\n(Base)",
    "GPT-4o\n(Reflexion)",
]

LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4"]

# Rows correspond to CONFIGS; columns correspond to LEVELS.
PASS_RATES = np.array(
    [
        [84, 72, 58, 40],
        [91, 84, 72, 61],
        [93, 86, 76, 64],
        [96, 91, 84, 74],
    ],
    dtype=float,
)

# Optional uncertainty, e.g. standard error or 95% CI half-width.
ERROR_BARS = np.array(
    [
        [3, 4, 5, 6],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [2, 2, 3, 4],
    ],
    dtype=float,
)

COLORS = {
    "Level 1": "#cfd5df",
    "Level 2": "#64d9ce",
    "Level 3": "#2ca02c",
    "Level 4": "#e43d30",
}


def plot_final_pass_rate() -> None:
    """Render and save the grouped bar chart."""
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

    x = np.arange(len(CONFIGS))
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for idx, level in enumerate(LEVELS):
        offset = (idx - (len(LEVELS) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            PASS_RATES[:, idx],
            width=bar_width,
            label=level,
            color=COLORS[level],
            edgecolor="white",
            linewidth=0.9,
            yerr=ERROR_BARS[:, idx],
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#3f4652"},
        )

    ax.set_ylabel("Final Pass Rate (%)")
    ax.set_xlabel("Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.set_axisbelow(True)

    ax.legend(
        title="Task Difficulty",
        ncol=len(LEVELS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
    )

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / "final_pass_rate_by_level.png"
    pdf_path = OUTPUT_DIR / "final_pass_rate_by_level.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    plot_final_pass_rate()
