"""用完整 MNIST 測試集，比較 10 個 seed 的單隱藏層 ONNX 模型準確率與交叉熵。"""

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_mnist(data_dir):
    """讀取本機 MNIST 測試圖片與正確答案。"""
    # MNIST 的 IDX 檔案開頭是格式資訊，不是圖片或標籤。
    # 圖片檔跳過前 16 bytes，標籤檔跳過前 8 bytes。
    images = np.fromfile(
        data_dir / "t10k-images-idx3-ubyte", dtype=np.uint8, offset=16
    )
    labels = np.fromfile(
        data_dir / "t10k-labels-idx1-ubyte", dtype=np.uint8, offset=8
    )

    # 確認讀到完整的 10,000 張測試圖片與答案。
    if images.size != 10000 * 28 * 28 or labels.size != 10000:
        raise ValueError("需要完整的 MNIST 測試集：10,000 張 28 × 28 圖片與標籤")

    # 每列是一張圖片，共 784 個像素。
    images = images.reshape(10000, 784)

    # 將像素從整數 0～255 轉成浮點數 0～1，與原本模型的用法一致。
    images = images.astype(np.float32) / 255.0
    return images, labels


def evaluate_model(model_path, images, labels):
    """逐張推論，支援本專案的固定輸入形狀及動態 batch 軸。"""
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)":
        raise ValueError(f"{model_path}: 需要單一 float32 圖片輸入")
    model_input = inputs[0]
    shape = list(model_input.shape)
    if shape and not isinstance(shape[0], int):
        shape[0] = 1
    if (not shape or shape[0] != 1
            or any(not isinstance(n, int) or n <= 0 for n in shape)
            or np.prod(shape) != 784):
        raise ValueError(f"{model_path}: 不支援的單張圖片形狀 {shape}")

    predictions = []
    losses = []
    for image, label in zip(images, labels):
        scores = session.run(None, {model_input.name: image.reshape(shape)})[0]
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != 10 or not np.isfinite(scores).all():
            raise ValueError(f"{model_path}: 輸出必須是 10 個有限 logits")
        predictions.append(int(scores.argmax()))
        # 穩定的 log-sum-exp，避免 logits 大時 exp 溢位。
        shifted = scores - scores.max()
        losses.append(np.log(np.exp(shifted).sum()) - shifted[int(label)])
    predictions = np.asarray(predictions)
    return int((predictions == labels).sum()), float(np.mean(losses))


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=folder.parent / "input")
    parser.add_argument("--data-dir", type=Path,
                        default=folder.parent / "data/MNIST/raw",
                        help="含 t10k IDX 檔案的 MNIST/raw 目錄")
    parser.add_argument("--output", type=Path,
                        help="選填：將各模型結果寫入指定 CSV；預設只顯示")
    args = parser.parse_args()

    if not all((args.data_dir / name).is_file() for name in
               ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")):
        parser.error(f"找不到完整 MNIST 測試集：{args.data_dir}；可用 --data-dir 指定")

    model_paths = [args.input_dir / f"mnist256x1_self{seed}" / "mnist_fcnn.onnx"
                   for seed in range(1, 11)]
    missing = [str(p) for p in model_paths if not p.is_file()]
    if missing:
        parser.error("缺少模型：" + ", ".join(missing))

    images, labels = load_mnist(args.data_dir)
    print(f"測試資料：{args.data_dir}；共 {len(labels)} 張；像素除以 255")
    results = []
    for seed, model_path in enumerate(model_paths, start=1):
        correct, loss = evaluate_model(model_path, images, labels)
        accuracy = correct / len(labels) * 100
        results.append({"model": model_path.parent.name, "seed": seed,
                        "correct": correct, "total": len(labels),
                        "accuracy_percent": accuracy, "cross_entropy": loss})
        print(f"Seed {seed:2d}: {correct}/{len(labels)} | "
              f"accuracy={accuracy:.2f}% | loss={loss:.6f}", flush=True)

    for key, title, unit in [
        ("accuracy_percent", "Accuracy", " percentage points"),
        ("cross_entropy", "Cross-entropy", ""),
    ]:
        values = np.array([row[key] for row in results])
        print(f"{title}: mean={values.mean():.6f}, "
              f"sample std={values.std(ddof=1):.6f}{unit}, "
              f"range=[{values.min():.6f}, {values.max():.6f}]")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"結果已儲存：{args.output}")


if __name__ == "__main__":
    main()
