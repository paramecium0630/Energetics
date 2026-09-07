import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1] # Path to the project directory
output_dir = project_dir / "output" / "256x6"
figure_dir = project_dir / "figure"
figure_dir.mkdir(exist_ok=True)

stability = pd.read_csv(output_dir / "shuffle_stability.csv")
energetics = pd.read_csv(output_dir / "shuffle_energetics.csv")
summary = pd.read_csv(output_dir / "shuffle_summary.csv")

# 清除欄名與文字內容可能存在的空白
stability.columns = stability.columns.str.strip()
energetics.columns = energetics.columns.str.strip()
summary.columns = summary.columns.str.strip()
stability["status"] = stability["status"].str.strip() #

df = stability.merge(
    energetics[["shuffle_id", "total_entropy"]],
    on="shuffle_id",
    how="left",
)

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
# axes[0, 0].axvline(
#     original_entropy,
#     color="red",
#     linestyle="--",
#     label="Original",
# )
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
