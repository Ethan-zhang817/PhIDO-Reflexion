"""Plot cumulative task success rate over retry rounds.

Use round 0 for the initial design attempt and rounds 1-3 for the three
allowed retries. Fill ``N_BY_LEVEL`` and cumulative success-count matrices; pass
rates are derived automatically.

Publication-style export (Nature figure skill): editable text in PDF/SVG, TIFF
for press. Outputs go under ``plot_statistics/results``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
FIG_STEM = "success_convergence_by_retry_round"
LABEL_SIZE = 8.0

PALETTE_NMI_PASTEL = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny": "#E4E4F0",
    "ours_large": "#F0C0CC",
    "delta_up": "#2E9E44",
    "neutral_dark": "#606060",
}

PALETTE_NATURE_MATERIAL = {
    "aqua": "#77D7D1",
    "teal": "#33B5A5",
    "lilac": "#B9A7E8",
    "violet": "#7C6CCF",
    "neutral": "#D9D9D9",
}

ROUNDS = np.array([0, 1, 2, 3])
ROUND_LABELS = ["初始", "重试 1", "重试 2", "重试 3"]
LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4"]

LEVEL_COLORS = [
    PALETTE_NATURE_MATERIAL["neutral"],
    PALETTE_NATURE_MATERIAL["aqua"],
    PALETTE_NATURE_MATERIAL["lilac"],
    PALETTE_NATURE_MATERIAL["violet"],
]

# One total trial count per level (same for both strategies).
N_BY_LEVEL = np.array([14, 7, 36, 5], dtype=float)

# Rows: LEVELS; columns: ROUNDS. Values are cumulative number of successful
# runs after allowing up to k retries. Replace these examples with measured data.
REFLEXION_SUCCESS_COUNTS_BY_LEVEL = np.array(
    [
        [11, 12, 13, 13],
        [4, 5, 5, 6],
        [13, 13, 16, 17],
        [0, 0, 0, 2],
    ],
    dtype=float,
)

RETRY_ONLY_SUCCESS_COUNTS_BY_LEVEL = np.array(
    [
        [11, 11, 12, 13],
        [4, 4, 5, 5],
        [13, 13, 13, 14],
        [0, 0, 0, 0],
    ],
    dtype=float,
)


def _success_rates_pct(success_counts: np.ndarray, n_by_level: np.ndarray) -> np.ndarray:
    """Return cumulative success rate (%) for each level and retry round."""
    if success_counts.shape != (len(LEVELS), len(ROUNDS)):
        raise ValueError(
            "success count matrix must have shape "
            f"({len(LEVELS)}, {len(ROUNDS)}); got {success_counts.shape}"
        )
    if success_counts.shape[0] != n_by_level.shape[0]:
        raise ValueError(
            "success count rows must match len(N_BY_LEVEL); "
            f"got {success_counts.shape[0]} vs {n_by_level.shape[0]}"
        )
    if np.any(success_counts > n_by_level[:, np.newaxis]):
        bad = np.argwhere(success_counts > n_by_level[:, np.newaxis])
        raise ValueError(
            "success counts must not exceed N_BY_LEVEL; "
            f"first violation at (level_idx, round_idx)={tuple(bad[0])}"
        )
    if np.any(np.diff(success_counts, axis=1) < 0):
        bad = np.argwhere(np.diff(success_counts, axis=1) < 0)
        raise ValueError(
            "success counts must be cumulative and non-decreasing across rounds; "
            f"first decrease starts at (level_idx, round_idx)={tuple(bad[0])}"
        )
    return np.divide(
        success_counts,
        n_by_level[:, np.newaxis],
        out=np.zeros_like(success_counts, dtype=float),
        where=n_by_level[:, np.newaxis] > 0,
    ) * 100.0


def _preferred_cjk_families() -> list[str]:
    """Pick one good CJK-capable sans first; keep Latin fonts for fallback chain."""
    latin = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
    scored: list[tuple[int, str]] = []
    for f in fm.fontManager.ttflist:
        name = f.name
        score = 0
        if "Noto" in name and "CJK" in name:
            score = 100
        elif "Source Han Sans" in name:
            score = 90
        elif "WenQuanYi" in name:
            score = 85
        elif name in ("SimHei", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB"):
            score = 75
        if score:
            scored.append((score, name))
    if scored:
        best = max(scored, key=lambda t: t[0])[1]
        return [best, *latin]
    return latin


def _apply_pub_rc() -> None:
    """Journal-oriented matplotlib defaults (editable vector text)."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _preferred_cjk_families(),
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": LABEL_SIZE,
            "legend.title_fontsize": LABEL_SIZE,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def _save_pub(fig: plt.Figure, stem: Path, *, dpi_raster: int = 600) -> None:
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        f"{stem}.png",
        dpi=dpi_raster,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        f"{stem}.tiff",
        dpi=dpi_raster,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def plot_success_convergence() -> None:
    """Render and save the convergence curve."""
    _apply_pub_rc()
    reflexion_success_by_level = _success_rates_pct(
        REFLEXION_SUCCESS_COUNTS_BY_LEVEL,
        N_BY_LEVEL,
    )
    retry_only_success_by_level = _success_rates_pct(
        RETRY_ONLY_SUCCESS_COUNTS_BY_LEVEL,
        N_BY_LEVEL,
    )

    fig_w_in = 183 / 25.4
    fig_h_in = fig_w_in * 0.46
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_w_in, fig_h_in),
        sharey=True,
        constrained_layout=False,
    )

    panels = [
        ("Reflexion", reflexion_success_by_level, "o"),
        ("Retry-only", retry_only_success_by_level, "s"),
    ]

    for ax, (strategy, success_by_level, marker) in zip(axes, panels):
        for level, color, values in zip(LEVELS, LEVEL_COLORS, success_by_level):
            ax.plot(
                ROUNDS,
                values,
                marker=marker,
                markersize=3.9,
                markeredgecolor="white",
                markeredgewidth=0.55,
                linewidth=1.35,
                color=color,
                label=level,
                zorder=4,
            )

        ax.set_title(
            strategy,
            fontsize=8.2,
            pad=5,
            color=PALETTE_NMI_PASTEL["neutral_dark"],
            fontweight="semibold",
        )
        ax.set_xlabel("允许重试轮数")
        ax.set_xticks(ROUNDS)
        ax.set_xticklabels(ROUND_LABELS)
        ax.set_xlim(-0.12, 3.12)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 25))
        ax.yaxis.grid(True, linestyle="-", linewidth=0.35, color="#000000", alpha=0.12)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", pad=3)
        ax.tick_params(axis="y", pad=2)
        ax.axvline(3, color=PALETTE_NMI_PASTEL["neutral_dark"], lw=0.45, alpha=0.35)
        ax.text(
            2.98,
            96,
            "上限",
            ha="right",
            va="top",
            fontsize=6.5,
            color=PALETTE_NMI_PASTEL["neutral_dark"],
        )

    axes[0].set_ylabel("累计成功率（%）")

    fig.suptitle(
        "任务成功率随迭代轮数收敛",
        fontsize=8.5,
        y=0.96,
        color=PALETTE_NMI_PASTEL["neutral_dark"],
        fontweight="semibold",
    )
    fig.text(
        0.99,
        0.925,
        "模型：GPT-5.1-codex",
        ha="right",
        va="center",
        fontsize=7.0,
        color=PALETTE_NMI_PASTEL["neutral_dark"],
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="任务难度",
        ncol=len(LEVELS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=0.9,
    )

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.24, top=0.82, wspace=0.18)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = OUTPUT_DIR / FIG_STEM
    _save_pub(fig, stem)
    plt.close(fig)

    for ext in ("svg", "pdf", "png", "tiff"):
        print(f"Saved: {stem}.{ext}")


if __name__ == "__main__":
    plot_success_convergence()
