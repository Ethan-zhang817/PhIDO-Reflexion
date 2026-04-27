"""Render a publication-style comparison table for pass-rate and cost results.

The table complements the grouped bar chart by replacing per-level detail with
compact summary metrics and cost-per-success comparisons.
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

# Replace with measured API cost per task, including all retry rounds actually
# consumed by each configuration. Unit: USD / task.
AVG_COST_PER_TASK = np.array([0.004, 0.011, 0.024, 0.052], dtype=float)


def _format_delta(value: float) -> str:
    if abs(value) < 1e-9:
        return "0.0"
    return f"{value:+.1f}"


def _build_table() -> tuple[list[str], list[list[str]], np.ndarray]:
    overall = PASS_RATES.mean(axis=1)
    hard_avg = PASS_RATES[:, 2:].mean(axis=1)
    lift_vs_mini_base = overall - overall[0]
    cost_per_success = AVG_COST_PER_TASK / np.maximum(overall / 100.0, 1e-9)

    columns = [
        "Configuration",
        "Overall\nPass",
        "Hard Pass\n(L3-L4)",
        "Lift vs\nMini Base",
        "Avg Cost\n/ Task",
        "Cost\n/ Success",
    ]

    values = np.column_stack(
        [overall, hard_avg, lift_vs_mini_base, AVG_COST_PER_TASK, cost_per_success]
    )
    table_text: list[list[str]] = []
    for config, row in zip(CONFIGS, values):
        table_text.append(
            [
                config,
                f"{row[0]:.1f}%",
                f"{row[1]:.1f}%",
                _format_delta(row[2]) + " pp",
                f"${row[3]:.3f}",
                f"${row[4]:.3f}",
            ]
        )

    return columns, table_text, values


def plot_comparison_table() -> None:
    """Render and save the comparison table."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )

    columns, table_text, values = _build_table()
    n_rows = len(table_text)
    n_cols = len(columns)

    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.axis("off")

    table = ax.table(
        cellText=table_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.75)

    header_color = "#1f4e79"
    config_col_color = "#eef3f8"
    summary_col_color = "#f7f3e8"
    edge_color = "#ffffff"

    for col in range(n_cols):
        cell = table[(0, col)]
        cell.set_facecolor(header_color)
        cell.set_edgecolor(edge_color)
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")

    # Pass-rate columns use a green gradient; cost columns use a reversed
    # orange gradient so lower cost reads as better.
    rate_min = float(values[:, :2].min())
    rate_max = float(values[:, :2].max())
    cost_min = float(values[:, 3:].min())
    cost_max = float(values[:, 3:].max())
    rate_cmap = plt.get_cmap("Greens")
    cost_cmap = plt.get_cmap("Oranges_r")

    for row in range(1, n_rows + 1):
        for col in range(n_cols):
            cell = table[(row, col)]
            cell.set_edgecolor(edge_color)

            if col == 0:
                cell.set_facecolor(config_col_color)
                cell.get_text().set_weight("bold")
                continue

            if col >= 3:
                cell.set_facecolor(summary_col_color)

            if 1 <= col <= 2:
                value = values[row - 1, col - 1]
                norm = (value - rate_min) / max(rate_max - rate_min, 1e-9)
                cell.set_facecolor(rate_cmap(0.18 + 0.45 * norm))

            if 4 <= col <= 5:
                value = values[row - 1, col - 1]
                norm = (value - cost_min) / max(cost_max - cost_min, 1e-9)
                cell.set_facecolor(cost_cmap(0.18 + 0.45 * norm))

            if col == 3 and values[row - 1, 2] > 0:
                cell.get_text().set_color("#0f766e")
                cell.get_text().set_weight("bold")

    ax.set_title(
        "Performance and Cost Comparison Across Model Configurations",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / "pass_rate_comparison_table.png"
    pdf_path = OUTPUT_DIR / "pass_rate_comparison_table.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    plot_comparison_table()
