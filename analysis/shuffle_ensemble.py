import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def histogram_bins(values, count=30):
    """Use one bin when the data span cannot reliably resolve count bins."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Histogram requires nonempty, finite data")
    lo, hi = values.min(), values.max()
    scale = max(1.0, abs(lo), abs(hi))
    if hi - lo <= 2 * count * np.spacing(scale):
        # Keep the original values; only widen the plotting interval.
        padding = 1e-12 * scale
        return np.array([lo - padding, hi + padding])
    return np.histogram_bin_edges(values, bins=count)

project_dir = Path(__file__).resolve().parents[1] # Path to the project directory
# Switch between "gaussian" and "shuffle". Paths can be overridden below.
data_mode = "gaussian"
if data_mode not in {"gaussian", "shuffle"}:
    raise ValueError("data_mode must be gaussian or shuffle")
# output_dir = project_dir / "output" / data_mode
# output_dir = project_dir / "shuffle_data" / "mnist256x1_linear_r1" / "layer"
output_dir = project_dir / "shuffle_data" / "mnist256x1_linear_r1" / "gaussian"

reference_dir = project_dir / "shuffle_data" / "mnist256x1_linear_r1" / "all"
plot_shuffle_comparison = False  # Independent, archived multi-case comparison
ensemble_label = "Gaussian samples" if data_mode == "gaussian" else "Shuffles"
# output_dir = project_dir / "shuffle_data" / "mnist256x1_linear" / "layer_fraction02"
figure_dir = project_dir / "figure"
figure_dir.mkdir(exist_ok=True)

original_layer_rates = None
layer_rates = None

summary_dir = reference_dir if data_mode == "gaussian" else output_dir
summary = pd.read_csv(summary_dir / "shuffle_summary.csv")
summary.columns = summary.columns.str.strip()

if data_mode == "gaussian" or (output_dir / "shuffle_trials.csv").exists():
    trials = pd.read_csv(output_dir / f"{data_mode}_trials.csv")
    layer_rates = pd.read_csv(output_dir / f"{data_mode}_layer_energetics.csv")
    node_layers = pd.read_csv(output_dir / "node_layers.csv")
    if trials["id"].duplicated().any() or node_layers["Node"].duplicated().any():
        raise ValueError("Duplicate shuffle or node ID")
    if layer_rates.duplicated(["id", "Layer"]).any():
        raise ValueError("Duplicate shuffle-layer result")
    expected_counts = node_layers.groupby("Layer").size().sort_index()
    if (trials["id"] <= 0).any() or (layer_rates["id"] < 0).any():
        raise ValueError("Shuffle trial IDs must be positive; only reference rows use id=0")
    original_layer_rates = layer_rates.loc[layer_rates["id"] == 0].copy()
    if data_mode == "gaussian":
        reference_rates = pd.read_csv(reference_dir / "shuffle_layer_energetics.csv")
        original_layer_rates = reference_rates.loc[reference_rates["id"] == 0].copy()
        if original_layer_rates.empty or original_layer_rates["Layer"].duplicated().any():
            raise ValueError("Gaussian reference requires one id=0 row per layer")
        parameters = pd.read_csv(output_dir / "gaussian_parameters.csv")
        if set(parameters["coupling_type"].str.strip()) != {str(summary.loc[0, "coupling_type"]).strip()}:
            raise ValueError("Gaussian and reference coupling types differ")
        reference_nodes = pd.read_csv(reference_dir / "node_layers.csv")
        if not node_layers.sort_values("Node").reset_index(drop=True).equals(
            reference_nodes.sort_values("Node").reset_index(drop=True)
        ):
            raise ValueError("Gaussian and reference node-layer mappings differ")
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
    if data_mode == "gaussian":
        stable_mask = df["stability"].str.strip().eq("stable")
        if not np.allclose(df.loc[stable_mask, "EPR_total"],
                           df.loc[stable_mask, "total_entropy"], rtol=1e-10, atol=1e-12):
            raise ValueError("Gaussian total EPR differs from the sum of layer EPR")
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
if data_mode == "shuffle":
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
    f"Mean {ensemble_label} total entropy production rate "
    f"(stable networks) = {mean_total_entropy:.15g}"
)
print(
    f"Std {ensemble_label} total entropy production rate "
    f"(stable networks) = {std_total_entropy:.15g}"
)

if data_mode == "gaussian":
    fig, (total_ax, stability_ax) = plt.subplots(1, 2, figsize=(12, 5))
    axes = np.array([[total_ax, None], [None, stability_ax]], dtype=object)
else:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total entropy production rate: stable networks only
axes[0, 0].hist(
    stable_df["total_entropy"].dropna(),
    bins=histogram_bins(stable_df["total_entropy"].dropna()),
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

if data_mode == "shuffle":
    # Minimum in-strength
    axes[0, 1].hist(
        df["min_kappa_in"],
        bins=histogram_bins(df["min_kappa_in"]),
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
        bins=histogram_bins(df["min_kappa_out"]),
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
    bins=histogram_bins(df["max_real_part"]),
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
    figure_dir / f"{data_mode}_ensemble_histograms.png",
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
        ax.hist(values, bins=histogram_bins(values), edgecolor="black", alpha=0.75,
                label=f"Stable {ensemble_label}")
        ax.axvline(row.original_EPR_total, color="red", linestyle="--",
                   linewidth=2, label="Original network")
        ax.set_title(f"Layer {row.Layer} (N={row.node_count})")
        ax.set_xlabel("Layer total EPR")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left", fontsize=8)
        std_text = f"{row.shuffle_std:.6g}" if pd.notna(row.shuffle_std) else "NA"
        ax.text(0.97, 0.75,
                f"Original = {row.original_EPR_total:.6g}\n"
                f"Sample mean = {row.shuffle_mean:.6g}\nSample std = {std_text}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85})
        ax.grid(alpha=0.2)
    for ax in list(layer_axes.flat)[len(layer_comparison):]:
        ax.set_visible(False)
    layer_fig.suptitle(f"Layer total EPR: {ensemble_label} versus original network")
    layer_fig.tight_layout()
    layer_fig.savefig(figure_dir / f"{data_mode}_layer_epr_histograms.png",
                      dpi=300, bbox_inches="tight")
    print(f"\nLayer total EPR comparison (stable {ensemble_label} only)")
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

if plot_shuffle_comparison:
    # One-off comparison of three GLOBAL shuffles and one LAYER shuffle.
    # Keep these paths explicit because this figure is only needed for this dataset.
    global_shuffle_dirs = {
        "Global 5%": project_dir / "shuffle_data" / "mnist256x1_linear_r1" / "all_fraction005",
        "Global 20%": project_dir / "shuffle_data" / "mnist256x1_linear_r1" / "all_fraction02",
        "Global 100%": project_dir / "shuffle_data" / "mnist256x1_linear_r1" / "all",
    }
    global_colors = {
        "Global 5%": "tab:blue",
        "Global 20%": "tab:orange",
        "Global 100%": "tab:green",
    }
    global_total_epr = {}
    global_layer_epr = {}
    comparison_original_layers = None

    for fraction_label, data_dir in global_shuffle_dirs.items():
        comparison_trials = pd.read_csv(data_dir / "shuffle_trials.csv")
        comparison_layers = pd.read_csv(data_dir / "shuffle_layer_energetics.csv")
        comparison_trials["stability"] = comparison_trials["stability"].str.strip()
        stable_ids = comparison_trials.loc[
            comparison_trials["stability"] == "stable", "id"
        ]

        reference_rows = comparison_layers.loc[comparison_layers["id"] == 0].copy()
        if comparison_original_layers is None:
            comparison_original_layers = reference_rows

        stable_layers = comparison_layers.loc[
            comparison_layers["id"].isin(stable_ids)
        ].copy()
        original_by_layer = reference_rows.set_index("Layer")["EPR_total"]
        stable_layers["EPR_change"] = (
            stable_layers["EPR_total"]
            - stable_layers["Layer"].map(original_by_layer)
        )
        n_layers = reference_rows["Layer"].nunique()
        total_epr = stable_layers.groupby("id")["EPR_total"].sum(
            min_count=n_layers
        ).dropna() - reference_rows["EPR_total"].sum()
        global_total_epr[fraction_label] = total_epr
        global_layer_epr[fraction_label] = stable_layers

    # Overlay the total-EPR changes using common bin edges.
    all_total_values = np.concatenate([values.to_numpy() for values in global_total_epr.values()])
    total_bins = histogram_bins(all_total_values)
    comparison_fig, comparison_ax = plt.subplots(figsize=(8, 6))
    for fraction_label, values in global_total_epr.items():
        comparison_ax.hist(
            values,
            bins=total_bins,
            alpha=0.4,
            linewidth=2,
            color=global_colors[fraction_label],
            label=f"{fraction_label} (mean={values.mean():.4g}, N={len(values)})",
        )
    comparison_ax.axvline(
        0.0,
        color="red",
        linestyle="--",
        linewidth=2,
        label=r"Original network ($\Delta EPR=0$)",
    )
    comparison_ax.set_xlabel(r"$\Delta EPR_{\mathrm{net}}$", fontsize=14)
    comparison_ax.set_ylabel("Count", fontsize=14)
    comparison_ax.set_title("Change in total EPR after shuffling")
    comparison_ax.legend(fontsize=12)
    comparison_ax.grid(alpha=0.2)
    comparison_fig.tight_layout()
    comparison_fig.savefig(
        figure_dir / "global_shuffle_epr_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Make the same EPR-change comparison separately for every layer.
    comparison_original_layers = comparison_original_layers.sort_values("Layer")
    n_comparison_layers = len(comparison_original_layers)
    ncols = min(3, n_comparison_layers)
    nrows = (n_comparison_layers + ncols - 1) // ncols
    comparison_layer_fig, comparison_layer_axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
    )
    for ax, reference_row in zip(
        comparison_layer_axes.flat,
        comparison_original_layers.itertuples(index=False),
    ):
        layer_values = {
            fraction_label: rows.loc[
                rows["Layer"] == reference_row.Layer, "EPR_change"
            ].dropna()
            for fraction_label, rows in global_layer_epr.items()
        }
        all_layer_values = np.concatenate(
            [values.to_numpy() for values in layer_values.values()]
        )
        layer_bins = histogram_bins(all_layer_values)
        for fraction_label, values in layer_values.items():
            ax.hist(
                values,
                bins=layer_bins,
                alpha=0.4,
                linewidth=2,
                color=global_colors[fraction_label],
                label=f"{fraction_label} (mean={values.mean():.4g})",
            )
        ax.axvline(
            0.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label=r"Original ($\Delta EPR=0$)",
        )
        ax.set_title(f"Layer {reference_row.Layer} (N={reference_row.node_count})")
        ax.set_xlabel(r"$\Delta EPR_{\mathrm{layer}}$", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.2)
    for ax in list(comparison_layer_axes.flat)[n_comparison_layers:]:
        ax.set_visible(False)
    comparison_layer_fig.suptitle("Change in layer EPR after shuffling")
    comparison_layer_fig.tight_layout()
    comparison_layer_fig.savefig(
        figure_dir / "global_shuffle_layer_epr_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    # plt.show()
