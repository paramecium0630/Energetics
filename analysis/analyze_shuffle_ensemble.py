import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1] # Path to the project directory
# output_dir = project_dir / "output" / "shuffle"
output_dir = project_dir / "shuffle_data" / "mnist256x1_diffusive"
figure_dir = project_dir / "figure"
figure_dir.mkdir(exist_ok=True)

original_layer_rates = None
layer_rates = None

summary = pd.read_csv(output_dir / "shuffle_summary.csv")
summary.columns = summary.columns.str.strip()

if (output_dir / "shuffle_trials.csv").exists():
    trials = pd.read_csv(output_dir / "shuffle_trials.csv")
    layer_rates = pd.read_csv(output_dir / "shuffle_layer_energetics.csv")
    node_layers = pd.read_csv(output_dir / "node_layers.csv")
    if trials["id"].duplicated().any() or node_layers["Node"].duplicated().any():
        raise ValueError("Duplicate shuffle or node ID")
    if layer_rates.duplicated(["id", "Layer"]).any():
        raise ValueError("Duplicate shuffle-layer result")
    expected_counts = node_layers.groupby("Layer").size().sort_index()
    if (trials["id"] <= 0).any() or (layer_rates["id"] < 0).any():
        raise ValueError("Shuffle trial IDs must be positive; only reference rows use id=0")
    original_layer_rates = layer_rates.loc[layer_rates["id"] == 0].copy()
    layer_rates = layer_rates.loc[layer_rates["id"] > 0].copy()
    if not original_layer_rates.empty:
        counts = original_layer_rates.set_index("Layer")["node_count"].sort_index()
        if not counts.equals(expected_counts):
            raise ValueError("Incomplete original node/layer counts")
        if not np.isfinite(original_layer_rates["EPR_total"]).all():
            raise ValueError("Original layer EPR contains NaN or infinity")
        if not np.isclose(original_layer_rates["EPR_total"].sum(),
                          summary.loc[0, "original_entropy"], rtol=1e-10, atol=1e-12):
            raise ValueError("Original layer EPR does not sum to summary original_entropy")
    if set(layer_rates["id"]) != set(trials["id"]):
        raise ValueError("Shuffle IDs differ between trials and layer results")
    for trial_id, rows in layer_rates.groupby("id"):
        counts = rows.set_index("Layer")["node_count"].sort_index()
        if not counts.equals(expected_counts):
            raise ValueError(f"Incomplete node/layer counts for shuffle {trial_id}")
    # Require every layer so an unstable trial's NaNs cannot become total EPR=0.
    totals = layer_rates.groupby("id")["EPR_total"].sum(min_count=len(expected_counts))
    df = trials.merge(totals.rename("total_entropy"), on="id", validate="one_to_one")
    df = df.rename(columns={"id": "shuffle_id", "stability": "status"})
else:
    # Read archived pre-migration ensembles without rewriting their data.
    stability = pd.read_csv(output_dir / "shuffle_stability.csv")
    energetics = pd.read_csv(output_dir / "shuffle_energetics.csv")
    stability.columns = stability.columns.str.strip()
    energetics.columns = energetics.columns.str.strip()
    df = stability.merge(
        energetics[["shuffle_id", "total_entropy"]],
        on="shuffle_id", how="left", validate="one_to_one",
    )
df["status"] = df["status"].str.strip()
if not np.isfinite(df.loc[df["status"] == "stable", "total_entropy"]).all():
    raise ValueError("Stable shuffle is missing a layer EPR result")

stable_df = df[df["status"] == "stable"].copy()
original_entropy = summary.loc[0, "original_entropy"]
original_min_kappa_in = summary.loc[0, "original_min_kappa_in"]
original_min_kappa_out = summary.loc[0, "original_min_kappa_out"]
original_max_real_part = summary.loc[0, "original_max_real_part"]

if stable_df.empty:
    raise ValueError("No stable shuffled network has an entropy result")


def add_statistics_text(ax, mean_value, std_value, x_position, alignment):
    std_label = f"{std_value:.6g}" if pd.notna(std_value) else "NA"
    ax.text(
        x_position,
        0.95,
        f"Mean = {mean_value:.6g}\nSample std = {std_label}",
        transform=ax.transAxes,
        horizontalalignment=alignment,
        verticalalignment="top",
        fontsize=13,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


mean_total_entropy = stable_df["total_entropy"].mean()
std_total_entropy = stable_df["total_entropy"].std(ddof=1)
mean_min_kappa_in = df["min_kappa_in"].mean()
std_min_kappa_in = df["min_kappa_in"].std(ddof=1)
mean_min_kappa_out = df["min_kappa_out"].mean()
std_min_kappa_out = df["min_kappa_out"].std(ddof=1)
mean_max_real_part = df["max_real_part"].mean()
std_max_real_part = df["max_real_part"].std(ddof=1)
print(
    r"Minimum \kappa^{in} of the original network "
    f" = {original_min_kappa_in:.15g}"
)
print(
    r"Minimum \kappa^{out} of the original network "
    f" = {original_min_kappa_out:.15g}"
)
print(
    r"Maximum \lambda of the original network "
    f" = {original_max_real_part:.15g}"
)
print(
    "Total entropy production rate of the original "
    f"network = {original_entropy:.15g}"
)
print(
    "Mean shuffled total entropy production rate "
    f"(stable networks) = {mean_total_entropy:.15g}"
)
print(
    "Std shuffled total entropy production rate "
    f"(stable networks) = {std_total_entropy:.15g}"
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total entropy production rate: stable networks only
axes[0, 0].hist(
    stable_df["total_entropy"].dropna(),
    bins=30,
    edgecolor="black",
)
# 同一組 shuffle_summary.csv 記錄的原始訓練網路總 EPR。
axes[0, 0].axvline(
    original_entropy,
    color="red",
    linestyle="--",
    label="Original trained network",
)
axes[0, 0].legend(loc="upper left")
# axes[0, 0].axvline(
#     mean_total_entropy,
#     color="green",
#     linestyle="--",
#     label="Shuffle mean",
# )
axes[0, 0].set_xlabel("Total entropy production rate", fontsize=18)
axes[0, 0].set_ylabel("Count", fontsize=18)
add_statistics_text(
    axes[0, 0], mean_total_entropy, std_total_entropy, 0.97, "right"
)

# Minimum in-strength
axes[0, 1].hist(
    df["min_kappa_in"],
    bins=30,
    edgecolor="black",
)
axes[0, 1].axvline(
    original_min_kappa_in,
    color="red",
    linestyle="--",
    label="Original",
)
axes[0, 1].set_xlabel(r"Minimum $\kappa^{in}$", fontsize=18)
axes[0, 1].set_ylabel("Count", fontsize=18)
axes[0, 1].legend()
add_statistics_text(
    axes[0, 1], mean_min_kappa_in, std_min_kappa_in, 0.03, "left"
)

# Minimum out-strength
axes[1, 0].hist(
    df["min_kappa_out"],
    bins=30,
    edgecolor="black",
)
axes[1, 0].axvline(
    original_min_kappa_out,
    color="red",
    linestyle="--",
    label="Original",
)
axes[1, 0].set_xlabel(r"Minimum $\kappa^{out}$", fontsize=18)
axes[1, 0].set_ylabel("Count", fontsize=18)
axes[1, 0].legend()
add_statistics_text(
    axes[1, 0], mean_min_kappa_out, std_min_kappa_out, 0.03, "left"
)

# Maximum real part of eigenvalues
axes[1, 1].hist(
    df["max_real_part"],
    bins=30,
    edgecolor="black",
)
axes[1, 1].axvline(
    0.0,
    color="black",
    linestyle=":",
    label="Stability boundary",
)
axes[1, 1].axvline(
    original_max_real_part,
    color="red",
    linestyle="--",
    label="Original",
)
axes[1, 1].set_xlabel(r"$\max\,\mathrm{Re}(\lambda(Q))$", fontsize=18)
axes[1, 1].set_ylabel("Count", fontsize=18)
axes[1, 1].legend()
add_statistics_text(
    axes[1, 1], mean_max_real_part, std_max_real_part, 0.97, "right"
)

fig.tight_layout()
fig.savefig(
    figure_dir / "shuffle_ensemble_histograms.png",
    dpi=300,
    bbox_inches="tight",
)

# 各層使用「層總 EPR」，只統計 stable trials，id=0 不加入樣本。
if original_layer_rates is not None and not original_layer_rates.empty:
    stable_layer_rates = layer_rates.loc[
        layer_rates["id"].isin(stable_df["shuffle_id"])
    ]
    layer_comparison = stable_layer_rates.groupby("Layer")["EPR_total"].agg(
        stable_count="size", shuffle_mean="mean", shuffle_std="std"
    ).reset_index()
    layer_comparison = layer_comparison.merge(
        original_layer_rates[["Layer", "node_count", "EPR_total"]].rename(
            columns={"EPR_total": "original_EPR_total"}
        ), on="Layer", validate="one_to_one",
    ).sort_values("Layer")
    layer_comparison["mean_minus_original"] = (
        layer_comparison["shuffle_mean"] - layer_comparison["original_EPR_total"]
    )
    # 不使用相對百分比，避免原始 EPR=0（例如輸入層）造成除零。
    ncols = min(3, len(layer_comparison))
    nrows = (len(layer_comparison) + ncols - 1) // ncols
    layer_fig, layer_axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
    )
    for ax, row in zip(layer_axes.flat, layer_comparison.itertuples(index=False)):
        values = stable_layer_rates.loc[
            stable_layer_rates["Layer"] == row.Layer, "EPR_total"
        ]
        ax.hist(values, bins=30, edgecolor="black", alpha=0.75,
                label="Stable shuffles")
        ax.axvline(row.original_EPR_total, color="red", linestyle="--",
                   linewidth=2, label="Original network")
        ax.set_title(f"Layer {row.Layer} (N={row.node_count})")
        ax.set_xlabel("Layer total EPR")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left", fontsize=8)
        std_text = f"{row.shuffle_std:.6g}" if pd.notna(row.shuffle_std) else "NA"
        ax.text(0.97, 0.75,
                f"Original = {row.original_EPR_total:.6g}\n"
                f"Shuffle mean = {row.shuffle_mean:.6g}\nSample std = {std_text}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85})
        ax.grid(alpha=0.2)
    for ax in list(layer_axes.flat)[len(layer_comparison):]:
        ax.set_visible(False)
    layer_fig.suptitle("Layer total EPR: stable shuffles versus original network")
    layer_fig.tight_layout()
    layer_fig.savefig(figure_dir / "shuffle_layer_epr_histograms.png",
                      dpi=300, bbox_inches="tight")
    print("\nLayer total EPR comparison (stable shuffles only)")
    print(layer_comparison.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
else:
    print("Layer EPR comparison unavailable: this dataset lacks original layer rates "
          "(id=0). Regenerate it with the updated Fortran program.")

# Stability versus total entropy production: stable networks only
# scatter_fig, scatter_ax = plt.subplots(figsize=(7, 6))

# scatter_ax.scatter(
#     stable_df["max_real_part"],
#     stable_df["total_entropy"],
#     s=35,
#     alpha=0.7,
#     edgecolors="black",
#     linewidths=0.4,
#     label="Shuffled networks",
# )
# # scatter_ax.scatter(
# #     original_max_real_part,
# #     original_entropy,
# #     s=140,
# #     color="red",
# #     marker="*",
# #     edgecolors="black",
# #     linewidths=0.6,
# #     zorder=3,
# #     label="Original network",
# # )
# scatter_ax.axvline(
#     0.0,
#     color="black",
#     linestyle=":",
#     label="Stability boundary",
# )

# scatter_ax.set_xlabel(r"$\max\,\mathrm{Re}(\lambda(Q))$", fontsize=18)
# scatter_ax.set_ylabel("Total entropy production rate", fontsize=18)
# scatter_ax.grid(alpha=0.25)
# scatter_ax.legend()

# scatter_fig.tight_layout()
# scatter_fig.savefig(
#     figure_dir / "stability_vs_entropy.png",
#     dpi=300,
#     bbox_inches="tight",
# )

plt.show()
