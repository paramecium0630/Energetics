"""繪製每輪 loss 與驗證準確率；不執行訓練。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 直接存圖，無需桌面視窗。
import matplotlib.pyplot as plt
import numpy as np


def plot_training(history, output_path, title="MNIST training history"):
    """history 每列為 epoch, train_loss, val_loss, val_accuracy (0~1)。"""
    data = np.asarray(history, dtype=float)
    epoch, train_loss, val_loss, accuracy = data.T
    best = int(np.argmin(val_loss))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].plot(epoch, train_loss, label="Training loss", color="tab:blue")
    axes[0].plot(epoch, val_loss, label="Validation loss", color="tab:orange")
    axes[0].scatter(epoch[best], val_loss[best], color="tab:red", zorder=3)
    axes[0].axvline(epoch[best], color="tab:red", linestyle="--", alpha=0.6,
                    label=f"Min validation loss: epoch {int(epoch[best])}")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss (lower is better)")
    axes[0].legend(fontsize=8)
    axes[1].plot(epoch, accuracy * 100, color="tab:green")
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("Validation accuracy (higher is better)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="training_history.csv 的路徑")
    args = parser.parse_args()
    history = np.loadtxt(args.csv, delimiter=",", skiprows=1, ndmin=2)
    destination = args.csv.with_name("training_curves.png")
    plot_training(history, destination)
    print(destination)
