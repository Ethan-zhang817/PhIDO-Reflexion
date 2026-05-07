"""Plot final pass rate by model configuration and task difficulty.

Fill ``N_BY_LEVEL`` (trials per difficulty, one per level) and ``PASS_COUNTS``
(rows = levels, columns = model configs); pass rates are derived automatically.

Publication-style export (Nature figure skill): editable text in PDF/SVG, TIFF
for press. Outputs go under ``plot_statistics/results``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
FIG_STEM = "final_pass_rate_by_level"
LABEL_SIZE = 8.0

# --- NMI pastel family (low saturation; difficulty encoded as intensity ramp) ---
PALETTE_NMI_PASTEL = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny": "#E4E4F0",
    "ours_base": "#E4CCD8",
    "ours_large": "#F0C0CC",
    "neutral_dark": "#606060",
}

CONFIGS = [
    "GPT-5.1-codex (Base)",
    "GPT-5.1-codex (Reflexion)",
    "GPT-5.3-codex (Base)",
    "GPT-5.3-codex (Reflexion)",
]

LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4"]

# Lighter → deeper hue for increasing difficulty (single unified family).
LEVEL_COLORS = [
    PALETTE_NMI_PASTEL["ours_tiny"],
    PALETTE_NMI_PASTEL["baseline_soft"],
    PALETTE_NMI_PASTEL["baseline_mid"],
    PALETTE_NMI_PASTEL["baseline_dark"],
]

# Total trials per level (row of PASS_COUNTS). Same across all configs in that row.
N_BY_LEVEL = np.array([14, 7, 36, 5], dtype=float)

# Rows: LEVELS; columns: CONFIGS. Success counts for that level under each model config.
PASS_COUNTS = np.array(
    [
        [11, 13, 12, 13],
        [4, 5, 5, 6],
        [13, 16, 17, 16],
        [0, 2, 2, 3],
    ],
    dtype=float,
)


def _pass_rates_pct(pass_counts: np.ndarray, n_by_level: np.ndarray) -> np.ndarray:
    """Return pass rate in %; shape (level, config) matching PASS_COUNTS."""
    n_level = len(LEVELS)
    n_cfg = len(CONFIGS)
    if pass_counts.shape != (n_level, n_cfg):
        raise ValueError(
            "PASS_COUNTS must have shape (len(LEVELS), len(CONFIGS)); "
            f"got {pass_counts.shape}, expected ({n_level}, {n_cfg})"
        )
    if n_by_level.shape[0] != n_level:
        raise ValueError(
            "N_BY_LEVEL length must match len(LEVELS); "
            f"got {n_by_level.shape[0]} vs {n_level}"
        )
    caps = n_by_level[:, np.newaxis]
    if np.any(pass_counts > caps):
        bad = np.argwhere(pass_counts > caps)
        raise ValueError(
            "PASS_COUNTS must not exceed N_BY_LEVEL for that row (level); "
            f"first violation at (level_row, config_col)={tuple(bad[0])}"
        )
    p = np.divide(
        pass_counts,
        caps,
        out=np.zeros_like(pass_counts, dtype=float),
        where=caps > 0,
    )
    return p * 100.0


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


def plot_final_pass_rate() -> None:
    """Render and save the grouped bar chart."""
    _apply_pub_rc()
    pass_rates = _pass_rates_pct(PASS_COUNTS, N_BY_LEVEL)

    # ~183 mm wide (double-column) × modest height, in inches.
    fig_w_in = 183 / 25.4
    # Enough bottom room for tilted model names, x-axis label, and legend.
    fig_h_in = fig_w_in * 0.50
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), constrained_layout=False)

    x = np.arange(len(CONFIGS))
    n_level = len(LEVELS)
    bar_width = 0.62 / max(n_level, 1)

    for idx, level in enumerate(LEVELS):
        offset = (idx - (n_level - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            pass_rates[idx, :],
            width=bar_width * 0.92,
            label=level,
            color=LEVEL_COLORS[idx],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

    ax.set_ylabel("最终通过率（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, rotation=18, ha="right", fontsize=7.5)
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.yaxis.grid(True, linestyle="-", linewidth=0.35, color="#000000", alpha=0.12)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=2)

    ax.set_title(
        "不同难度任务下的设计成功率（最终通过）",
        fontsize=8.5,
        pad=8,
        color=PALETTE_NMI_PASTEL["neutral_dark"],
        fontweight="semibold",
    )

    legend_handles = [
        Patch(facecolor="none", edgecolor="none", label="任务难度"),
        *[
            Patch(facecolor=color, edgecolor="white", linewidth=0.45, label=level)
            for level, color in zip(LEVELS, LEVEL_COLORS)
        ],
    ]
    leg = fig.legend(
        handles=legend_handles,
        ncol=n_level + 1,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        columnspacing=0.9,
        handletextpad=0.35,
    )

    fig.text(
        0.5,
        0.155,
        "模型配置",
        ha="center",
        va="center",
        fontsize=LABEL_SIZE,
        color="black",
    )

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.34, top=0.86)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / FIG_STEM
    _save_pub(fig, stem)
    plt.close(fig)

    for ext in ("svg", "pdf", "png", "tiff"):
        print(f"Saved: {stem}.{ext}")


if __name__ == "__main__":
    plot_final_pass_rate()
