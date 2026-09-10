"""Analyze theoretical entropy production for every FCNN layer."""

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
    / "output" / "256x6"
    / "energetics_theory_by_node_and_layer.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figure"

TOTAL_FIGURE_FILE = (
    FIGURE_DIR
    / "entropy_production_by_layer.png"
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
    "entropy_rate",
]

for column in required_columns:
    if column not in energetics.columns:
        raise ValueError(
            f"Missing column '{column}' in {INPUT_FILE.name}"
        )

if energetics["node"].duplicated().any():
    raise ValueError("The same node appears more than once")

if (energetics["layer"] < 1).any():
    raise ValueError("Layer IDs must be positive")

entropy_values = energetics["entropy_rate"].to_numpy()

if not np.all(np.isfinite(entropy_values)):
    raise ValueError("Column 'entropy_rate' contains NaN or infinity")

# -----------------------------------------------------------------------------
# 4. 按 layer 和 node 排序
# -----------------------------------------------------------------------------

energetics = energetics.sort_values(
    by=["layer", "node"]
).reset_index(drop=True)

layer_numbers = sorted(
    energetics["layer"].unique()
)

print(layer_numbers)

last_layer = max(layer_numbers)
first_layer = min(layer_numbers)

layer_labels = []

for layer in layer_numbers:
    if layer == first_layer:
        label = "Input"
    elif layer == last_layer:
        label = "Output"
    else:
        label = f"Hidden {layer - first_layer}"

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

    layer_values = layer_data["entropy_rate"]
    summary_row["entropy_total"] = layer_values.sum()
    summary_row["entropy_mean"] = layer_values.mean()

    if node_count > 1:
        summary_row["entropy_std"] = layer_values.std(ddof=1)
    else:
        summary_row["entropy_std"] = 0.0

    layer_summary_rows.append(summary_row)

layer_summary = pd.DataFrame(layer_summary_rows)

total_entropy = layer_summary["entropy_total"].sum()

if np.isclose(total_entropy, 0.0, rtol=0.0, atol=1.0e-14):
    raise ValueError("Total entropy production is zero; ratios are undefined")

layer_summary["entropy_fraction"] = (
    layer_summary["entropy_total"] / total_entropy
)
layer_summary["entropy_percentage"] = (
    100.0 * layer_summary["entropy_fraction"]
)

print("Layer-wise entropy-production statistics")
print(layer_summary.to_string(index=False))
print(f"Total entropy production rate = {total_entropy:.15g}")

# -----------------------------------------------------------------------------
# 6. 檢查 layer totals
# -----------------------------------------------------------------------------

total_from_nodes = energetics["entropy_rate"].sum()

if not np.isclose(
    total_from_nodes,
    total_entropy,
    rtol=1.0e-12,
    atol=1.0e-14,
):
    raise ValueError(
        "Layer totals do not reproduce the network total entropy"
    )

if not np.isclose(
    layer_summary["entropy_fraction"].sum(),
    1.0,
    rtol=1.0e-12,
    atol=1.0e-14,
):
    raise ValueError("Layer entropy fractions do not sum to one")


# -----------------------------------------------------------------------------
# 7. 畫每層占比與 node-wise entropy distribution
# -----------------------------------------------------------------------------

figure, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
)

percentage_bars = axes[0].bar(
    layer_labels,
    layer_summary["entropy_percentage"],
    color="tab:purple",
    alpha=0.8,
)
axes[0].set_title("Percentage of total entropy production")
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Percentage (%)")
axes[0].grid(axis="y", alpha=0.25)
axes[0].bar_label(percentage_bars, fmt="%.2f%%", padding=3)

entropy_values_by_layer = []

for layer in layer_numbers:
    layer_entropy_values = energetics.loc[
        energetics["layer"] == layer,
        "entropy_rate",
    ]
    entropy_values_by_layer.append(layer_entropy_values.to_numpy())

entropy_boxplot = axes[1].boxplot(
    entropy_values_by_layer,
    labels=layer_labels,
    patch_artist=True,
    showfliers=True,
    flierprops={
        "markersize": 2,
        "alpha": 0.3,
    },
)

for box in entropy_boxplot["boxes"]:
    box.set_facecolor("tab:orange")
    box.set_alpha(0.6)

axes[1].axhline(0.0, color="black", linewidth=1.0)
axes[1].set_title("Entropy production rate of each layer")
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Entropy production rate of each node")
axes[1].grid(axis="y", alpha=0.25)

figure.suptitle(
    f"Theoretical entropy production (total = {total_entropy:.6g})",
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
    f"Saved entropy figure: "
    f"{TOTAL_FIGURE_FILE}"
)

plt.show()
