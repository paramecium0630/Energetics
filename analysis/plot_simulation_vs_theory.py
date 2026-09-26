"""Plot simulation-versus-theory comparisons produced by Energetics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from energetics_io import read_node_energetics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURE_DIR = PROJECT_ROOT / "figure"
FIGURE_FILE = FIGURE_DIR / "simulation_theory_comparison.png"

def require_columns(data, columns, filename):
    """Raise a readable error if an output file has an old schema."""
    missing = [column for column in columns if column not in data.columns]

    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"{filename} is missing columns: {missing_text}. "
            "Re-run the updated Fortran program to regenerate the output."
        )


def plot_comparison(
    ax,
    theory_values,
    simulation_values,
    title,
    theory_label,
    simulation_label,
    color,
    max_points=None,
):
    """Draw an equal-axis comparison and report errors from all values."""
    theory_values = np.asarray(theory_values, dtype=float)
    simulation_values = np.asarray(simulation_values, dtype=float)

    finite = np.isfinite(theory_values) & np.isfinite(simulation_values)
    x_all = theory_values[finite]
    y_all = simulation_values[finite]

    if x_all.size == 0:
        raise ValueError(f"No finite values are available for {title}")

    error = y_all - x_all
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    x_plot = x_all
    y_plot = y_all

    if max_points is not None and x_all.size > max_points:
        rng = np.random.default_rng(12345)
        selected = rng.choice(x_all.size, size=max_points, replace=False)
        x_plot = x_all[selected]
        y_plot = y_all[selected]

    lower = min(np.min(x_all), np.min(y_all))
    upper = max(np.max(x_all), np.max(y_all))
    padding = 0.05 * (upper - lower)

    if padding == 0.0:
        padding = max(1.0e-12, 0.05 * max(1.0, abs(lower)))

    lower -= padding
    upper += padding

    ax.scatter(
        x_plot,
        y_plot,
        s=20,
        color=color,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(theory_label, fontsize=14)
    ax.set_ylabel(simulation_label, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.25)
    ax.text(
        0.04,
        0.96,
        f"RMSE = {rmse:.3e}\nmax |error| = {max_error:.3e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    print(
        f"{title}: n={x_all.size}, RMSE={rmse:.6e}, "
        f"max|error|={max_error:.6e}"
    )


# -----------------------------------------------------------------------------
# Read and validate output
# -----------------------------------------------------------------------------

simulation = read_node_energetics(OUTPUT_DIR / "energetics_simulation.csv")
theory = read_node_energetics(OUTPUT_DIR / "energetics_theory.csv")

energetics_fields = [
    "heat_rate",
    "entropy_rate",
    "work_rate",
    "internal_rate",
]

require_columns(simulation, ["node", *energetics_fields], "energetics_simulation.csv")
require_columns(theory, ["node", *energetics_fields], "energetics_theory.csv")
energetics = theory[["node", "layer", *energetics_fields]].merge(
    simulation[["node", "layer", *energetics_fields]],
    on=["node", "layer"],
    suffixes=("_theory", "_simulation"),
    validate="one_to_one",
)

if len(energetics) != len(theory) or len(energetics) != len(simulation):
    raise ValueError("Theory and simulation node/layer IDs do not match")

# Only energetics is written by the current Fortran output flow.
figure, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_settings = [
    ("heat_rate", "Heat rate", "tab:red"),
    ("entropy_rate", "Entropy production rate", "tab:orange"),
    ("work_rate", "Work rate", "tab:blue"),
    ("internal_rate", "Internal-energy rate", "tab:green"),
]

for ax, (field, title, color) in zip(axes.flat[:4], plot_settings):
    plot_comparison(
        ax,
        energetics[f"{field}_theory"],
        energetics[f"{field}_simulation"],
        title,
        "Theory",
        "Simulation",
        color,
    )

figure.suptitle("Simulation versus theory", fontsize=16)
figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
figure.savefig(FIGURE_FILE, dpi=200, bbox_inches="tight")

print(f"Saved comparison figure: {FIGURE_FILE}")

plt.show()
