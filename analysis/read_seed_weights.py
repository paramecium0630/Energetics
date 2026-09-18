"""讀取 10 個 seed 的 FCNN 權重，每段連接分別存成 L1、L2、L3 等。

執行：python read_seed_weights.py
在下方修改 HIDDEN_LAYERS 即可切換隱藏層數；每個隱藏層寬度為 256。
匯入：weights = load_seed_weights()
      W1 = weights[1]["L1"]  # seed 1，形狀 (256, 784)
      W2 = weights[1]["L2"]  # seed 1 的第二段連接
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
HIDDEN_LAYERS = 2  # 1：784→256→10；2：784→256→256→10


def load_weight_matrices(path, layer_sizes=(784, 256, 10)):
    """DAT 每行為 target source weight；矩陣排列為 [目標, 來源]。"""
    records = np.loadtxt(path, comments="#", ndmin=2)
    edge_count = sum(a * b for a, b in zip(layer_sizes[:-1], layer_sizes[1:]))
    if records.shape != (edge_count, 3):
        raise ValueError(f"{path}: 需要 {edge_count} 行、3 欄的權重資料")
    if not np.isfinite(records).all():
        raise ValueError(f"{path}: 資料包含 NaN 或 Inf")
    indices = records[:, :2]
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError(f"{path}: 節點編號必須是整數")

    # 每層的全域節點連續編號，由層寬自動算出範圍。
    source = records[:, 1]
    target = records[:, 0]
    offsets = np.concatenate([[0], np.cumsum(layer_sizes)])
    specifications = {
        f"L{i + 1}": (offsets[i] + 1, offsets[i + 1],
                      offsets[i + 1] + 1, offsets[i + 2])
        for i in range(len(layer_sizes) - 1)
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


def load_seed_weights(input_dir=INPUT_DIR, hidden_layers=HIDDEN_LAYERS):
    """回傳 weights[seed][layer]；seed 為整數 1~10。"""
    if hidden_layers < 1:
        raise ValueError("hidden_layers 必須至少為 1")
    layer_sizes = [784] + [256] * hidden_layers + [10]
    input_dir = Path(input_dir)
    weights = {}
    for seed in range(1, 11):
        path = input_dir / f"mnist256x{hidden_layers}_self{seed}" / "weighted_matrix.dat"
        weights[seed] = load_weight_matrices(path, layer_sizes)
    return weights


def main():
    weights = load_seed_weights(INPUT_DIR, HIDDEN_LAYERS)
    layers = list(weights[1])
    n_layers = len(layers)

    for layer in layers:
        print(f"{layer}: {weights[1][layer].shape} [target, source]")
    headers = ["Seed"]
    for layer in layers:
        headers.extend([f"Mean weight ({layer})", f"Std weight ({layer})"])
    print("\t".join(headers))
    for seed in range(1, 11):
        row = [str(seed)]
        for layer in layers:
            w = weights[seed][layer]
            row.extend([f"{w.mean():.10g}", f"{w.std(ddof=1):.10g}"])
        print("\t".join(row))

    # 每層一張子圖，10 個 seed 使用相同的分箱邊界。
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4.5))
    for ax, layer in zip(axes, layers):
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
    fig_fit, axes_fit = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4.5))
    for ax, layer in zip(axes_fit, layers):
        values = np.concatenate([weights[seed][layer].ravel()
                                 for seed in range(1, 11)])
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
        matrices = list(weights[seed].values())
        # 列加總：每段的目標節點，排除沒有入邊的輸入層。
        in_degrees[seed] = np.concatenate([w.sum(axis=1) for w in matrices])
        # 欄加總：每段的來源節點，排除沒有出邊的輸出層。
        out_degrees[seed] = np.concatenate([w.sum(axis=0) for w in matrices])
        # 不刪除因正負權重抵消而恰好為零的數值。

    fig_degree, axes_degree = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, degrees, title in zip(
        axes_degree, [in_degrees, out_degrees],
        [f"Weighted in-degree ({len(in_degrees[1])} nodes/seed)",
         f"Weighted out-degree ({len(out_degrees[1])} nodes/seed)"],
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
