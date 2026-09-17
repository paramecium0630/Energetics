"""讀取 10 個 seed 的權重，分成 L1 (784→256) 與 L2 (256→10)。

執行：python read_seed_weights.py
匯入：weights = load_seed_weights()
      W1 = weights[1]["L1"]  # seed 1，形狀 (256, 784)
      W2 = weights[1]["L2"]  # seed 1，形狀 (10, 256)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = Path(__file__).resolve().parent.parent / "input"


def load_weight_matrices(path):
    """DAT 每行為 target source weight；矩陣排列為 [目標, 來源]。"""
    records = np.loadtxt(path, comments="#", ndmin=2)
    if records.shape != (784 * 256 + 256 * 10, 3):
        raise ValueError(f"{path}: 需要 203264 行、3 欄的權重資料")
    if not np.isfinite(records).all():
        raise ValueError(f"{path}: 資料包含 NaN 或 Inf")
    indices = records[:, :2]
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError(f"{path}: 節點編號必須是整數")

    # 全域節點：輸入 1~784，隱藏 785~1040，輸出 1041~1050。
    source = records[:, 1]
    target = records[:, 0]
    specifications = {
        "L1": (1, 784, 785, 1040),
        "L2": (785, 1040, 1041, 1050),
    }
    matrices = {}
    for layer, (source_first, source_last, target_first, target_last) in specifications.items():
        mask = ((source >= source_first) & (source <= source_last)
                & (target >= target_first) & (target <= target_last))
        edges = records[mask]
        n_in = source_last - source_first + 1
        n_out = target_last - target_first + 1
        if len(edges) != n_in * n_out:
            raise ValueError(f"{path}: {layer} 的連接數量或節點範圍不正確")
        row = (edges[:, 0] - target_first).astype(int)
        col = (edges[:, 1] - source_first).astype(int)
        if np.unique(row * n_in + col).size != n_in * n_out:
            raise ValueError(f"{path}: {layer} 有重複或缺少的連接")
        # 按節點編號填入，因此 DAT 的行順序可以不同。
        matrix = np.empty((n_out, n_in), dtype=np.float64)
        matrix[row, col] = edges[:, 2]
        matrices[layer] = matrix
    return matrices


def load_seed_weights(input_dir=INPUT_DIR):
    """回傳 weights[seed]['L1' 或 'L2']；seed 為整數 1~10。"""
    input_dir = Path(input_dir)
    weights = {}
    for seed in range(1, 11):
        path = input_dir / f"mnist256x1_self{seed}" / "weighted_matrix.dat"
        weights[seed] = load_weight_matrices(path)
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    args = parser.parse_args()
    weights = load_seed_weights(args.input_dir)

    # 第一維按照 seed 1~10 排列，保留每層矩陣形狀。
    layer1_weights = np.stack([weights[seed]["L1"] for seed in range(1, 11)])
    layer2_weights = np.stack([weights[seed]["L2"] for seed in range(1, 11)])
    print(f"L1 (784 -> 256): {layer1_weights.shape} [seed, target, source]")
    print(f"L2 (256 -> 10):  {layer2_weights.shape} [seed, target, source]")
    print("Seed\tMean weight (L1)\tStd weight (L1)\tMean weight (L2)\tStd weight (L2)")
    for seed in range(1, 11):
        w1, w2 = weights[seed]["L1"], weights[seed]["L2"]
        # 與 analyze_fcnn_inputs.py 一致：樣本標準差 ddof=1。
        print(f"{seed}\t{w1.mean():.10g}\t{w1.std(ddof=1):.10g}"
              f"\t{w2.mean():.10g}\t{w2.std(ddof=1):.10g}")

    # 每層一張子圖，10 個 seed 使用相同的分箱邊界。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, layer in zip(axes, ["L1", "L2"]):
        all_values = np.concatenate([weights[seed][layer].ravel()
                                     for seed in range(1, 11)])
        bins = np.linspace(all_values.min(), all_values.max(), 81)
        for seed in range(1, 11):
            ax.hist(weights[seed][layer].ravel(), bins=bins,
                    density=True, histtype="step", label=f"Seed {seed}")
        ax.set_title(f"Weight distribution ({layer})")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
    fig.tight_layout()

    # 第二張圖：將 10 個 seed 的權重合併，每層各擬合一條 Laplace 曲線。
    fig_fit, axes_fit = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, layer, values in zip(
        axes_fit, ["L1", "L2"],
        [layer1_weights.ravel(), layer2_weights.ravel()],
    ):
        # 沿用 analyze_fcnn_inputs.py 的平均值中心估計，非自由位置 MLE。
        mu = values.mean()
        b = np.mean(np.abs(values - mu))
        ax.hist(values, bins=80, density=True, alpha=0.5,
                label="All 10 seeds")
        if np.ptp(values) > 0 and b > 0:
            x = np.sort(np.append(np.linspace(values.min(), values.max(), 1000), mu))
            pdf = np.exp(-np.abs(x - mu) / b) / (2 * b)
            ax.plot(x, pdf, color="red", linewidth=2,
                    label=fr"Laplace: $\mu={mu:.4g},\ b={b:.4g}$")
            print(f"Pooled {layer}: N={values.size}, mu={mu:.8g}, b={b:.8g}")
        else:
            ax.text(0.5, 0.8, "Constant weights: no Laplace fit",
                    transform=ax.transAxes, ha="center")
        ax.set_title(f"Pooled weights + Laplace ({layer})")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig_fit.tight_layout()

    # 第三張圖：weighted degree 是帶正負號的權重總和，不包含 bias。
    in_degrees = {}
    out_degrees = {}
    for seed in range(1, 11):
        w1, w2 = weights[seed]["L1"], weights[seed]["L2"]
        # 矩陣為 [target, source]，每列加總是入強度，每欄加總是出強度。
        # 入強度只取隱藏層 + 輸出層，排除輸入層的 784 個結構性零值。
        in_degrees[seed] = np.concatenate([w1.sum(axis=1), w2.sum(axis=1)])
        # 出強度只取輸入層 + 隱藏層，排除輸出層的 10 個結構性零值。
        out_degrees[seed] = np.concatenate([w1.sum(axis=0), w2.sum(axis=0)])
        # 不刪除因正負權重抵消而恰好為零的數值。

    fig_degree, axes_degree = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, degrees, title in zip(
        axes_degree, [in_degrees, out_degrees],
        ["Weighted in-degree (266 nodes/seed)",
         "Weighted out-degree (1040 nodes/seed)"],
    ):
        all_values = np.concatenate(list(degrees.values()))
        bins = np.histogram_bin_edges(all_values, bins=60)
        for seed in range(1, 11):
            ax.hist(degrees[seed], bins=bins, density=True,
                    histtype="step", label=f"Seed {seed}")
        ax.set_title(title)
        ax.set_xlabel("Signed sum of weights")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
    fig_degree.tight_layout()

    # 第四張圖：合併全部 seed 的 weighted degree（不是取平均）。
    fig_pooled, axes_pooled = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, degrees, title in zip(
        axes_pooled, [in_degrees, out_degrees],
        ["Pooled weighted in-degree", "Pooled weighted out-degree"],
    ):
        values = np.concatenate(list(degrees.values()))
        ax.hist(values, bins=60, density=True, alpha=0.6,
                label=f"All 10 seeds (N={values.size})")
        ax.set_title(title)
        ax.set_xlabel("Signed sum of weights")
        ax.set_ylabel("Probability density")
        ax.legend()
        ax.grid(alpha=0.2)
    fig_pooled.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
