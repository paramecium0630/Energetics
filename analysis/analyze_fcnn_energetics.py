"""Plot theoretical energetic rates for every FCNN layer."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. 設定檔案位置
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "energetics_theory_by_node_and_layer.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figure"

TOTAL_FIGURE_FILE = (
    FIGURE_DIR
    / "energetics_theory_layer_totals.png"
)

DISTRIBUTION_FIGURE_FILE = (
    FIGURE_DIR
    / "energetics_theory_layer_distributions.png"
)

# -----------------------------------------------------------------------------
# 2. 讀取資料
# -----------------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. "
        "Run the Fortran program with an FCNN first."
    )

# 第一行是文字標題，所以使用 skiprows=1。
energetics = pd.read_csv(INPUT_FILE, skiprows=1)

# 移除欄位名稱前後可能存在的空白。
energetics.columns = energetics.columns.str.strip()

# -----------------------------------------------------------------------------
# 3. 檢查必要欄位
# -----------------------------------------------------------------------------

required_columns = [
    "layer",
    "node",
    "heat_rate",
    "entropy_rate",
    "work_rate",
    "internal_rate",
]

for column in required_columns:
    if column not in energetics.columns:
        raise ValueError(
            f"Missing column '{column}' in {INPUT_FILE.name}"
        )

if energetics["node"].duplicated().any():
    raise ValueError("The same node appears more than once")

if (energetics["layer"] < 0).any():
    raise ValueError("Layer IDs must be non-negative")

rate_columns = [
    "heat_rate",
    "entropy_rate",
    "work_rate",
    "internal_rate",
]

for column in rate_columns:
    values = energetics[column].to_numpy()

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Column '{column}' contains NaN or infinity"
        )

# -----------------------------------------------------------------------------
# 4. 按 layer 和 node 排序
# -----------------------------------------------------------------------------

energetics = energetics.sort_values(
    by=["layer", "node"]
).reset_index(drop=True)

layer_numbers = sorted(
    energetics["layer"].unique()
)

last_layer = max(layer_numbers)

layer_labels = []

for layer in layer_numbers:
    if layer == 0:
        label = "Input"
    elif layer == last_layer:
        label = "Output"
    else:
        label = f"Hidden {layer}"

    layer_labels.append(label)

# -----------------------------------------------------------------------------
# 5. 計算每一層的統計量
# -----------------------------------------------------------------------------

layer_summary_rows = []

for layer in layer_numbers:

    # 只選擇目前這一層的節點。
    layer_data = energetics[
        energetics["layer"] == layer
    ]

    node_count = len(layer_data)

    summary_row = {
        "layer": layer,
        "node_count": node_count,
    }

    for rate in rate_columns:
        layer_values = layer_data[rate]

        summary_row[f"{rate}_total"] = (
            layer_values.sum()
        )

        summary_row[f"{rate}_mean"] = (
            layer_values.mean()
        )

        if node_count > 1:
            summary_row[f"{rate}_std"] = (
                layer_values.std()
            )
        else:
            summary_row[f"{rate}_std"] = 0.0

    layer_summary_rows.append(summary_row)

layer_summary = pd.DataFrame(layer_summary_rows)

print("Layer-wise energetic statistics")
print(layer_summary)

# -----------------------------------------------------------------------------
# 6. 檢查 layer totals
# -----------------------------------------------------------------------------

for rate in rate_columns:

    total_from_nodes = energetics[rate].sum()

    total_from_layers = layer_summary[
        f"{rate}_total"
    ].sum()

    if not np.isclose(
        total_from_nodes,
        total_from_layers,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError(
            f"Layer totals do not reproduce the "
            f"network total for {rate}"
        )


# -----------------------------------------------------------------------------
# 7. 畫每層 total energetic rates
# -----------------------------------------------------------------------------

plot_settings = [
    ("heat_rate", "Heat rate", "tab:red"),
    (
        "entropy_rate",
        "Entropy production rate",
        "tab:orange",
    ),
    ("work_rate", "Work rate", "tab:blue"),
    (
        "internal_rate",
        "Internal-energy rate",
        "tab:green",
    ),
]

figure, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
)

for ax, setting in zip(
    axes.flat,
    plot_settings,
):
    rate_name = setting[0]
    title = setting[1]
    color = setting[2]

    total_column = f"{rate_name}_total"

    ax.bar(
        layer_labels,
        layer_summary[total_column],
        color=color,
        alpha=0.8,
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total rate")
    ax.grid(axis="y", alpha=0.25)

figure.suptitle(
    "Theoretical energetic rates by layer",
    fontsize=16,
)

figure.tight_layout(
    rect=(0.0, 0.0, 1.0, 0.96)
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

figure.savefig(
    TOTAL_FIGURE_FILE,
    dpi=200,
    bbox_inches="tight",
)

print(
    f"Saved total-rate figure: "
    f"{TOTAL_FIGURE_FILE}"
)

# -----------------------------------------------------------------------------
# 8. 畫每層內部的 node-wise distribution
# -----------------------------------------------------------------------------

figure, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
)

for ax, setting in zip(
    axes.flat,
    plot_settings,
):
    rate_name = setting[0]
    title = setting[1]
    color = setting[2]

    values_by_layer = []

    for layer in layer_numbers:
        layer_values = energetics.loc[
            energetics["layer"] == layer,
            rate_name,
        ]

        values_by_layer.append(
            layer_values.to_numpy()
        )

    boxplot = ax.boxplot(
        values_by_layer,
        tick_labels=layer_labels,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "markersize": 2,
            "alpha": 0.3,
        },
    )

    for box in boxplot["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.6)

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rate per node")
    ax.grid(axis="y", alpha=0.25)

figure.suptitle(
    "Distribution of theoretical node rates in each layer",
    fontsize=16,
)

figure.tight_layout(
    rect=(0.0, 0.0, 1.0, 0.96)
)

figure.savefig(
    DISTRIBUTION_FIGURE_FILE,
    dpi=200,
    bbox_inches="tight",
)

print(
    f"Saved distribution figure: "
    f"{DISTRIBUTION_FIGURE_FILE}"
)

plt.show()