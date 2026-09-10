"""將目前 FCNN ONNX 的全連接層輸出為 Energetics 的 DAT 格式。"""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

# 路徑相對於此程式，從其他目錄執行也能找到模型。
base_dir = Path(__file__).resolve().parent.parent

model = onnx.load(base_dir / "output/mnist_fcnn.onnx")
parameters = {
    tensor.name: numpy_helper.to_array(tensor)
    for tensor in model.graph.initializer
}

# 按運算圖中的層順序取得 Gemm 參數，避免把 Reshape 常數當成權重。
# 統一成 weight[目標神經元, 來源神經元]。
layers = []
for node in model.graph.node:
    
    if node.op_type != "Gemm":
        continue
    if len(node.input) != 3 or any(
        name not in parameters for name in node.input[1:]
    ):
        raise ValueError("目前只支援具有固定 weight 和 bias 的 Gemm 層")
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attrs.get("transA", 0) != 0:
        raise ValueError("不支援 transA 非零的 Gemm 層")
    weight = parameters[node.input[1]].astype(np.float64)
    if not attrs.get("transB", 0):
        weight = weight.T
    weight = weight * attrs.get("alpha", 1.0)
    bias = parameters[node.input[2]].astype(np.float64) * attrs.get("beta", 1.0)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError(f"不支援的參數形狀：{node.name}")
    if layers and weight.shape[1] != layers[-1][0].shape[0]:
        raise ValueError("相鄰全連接層的維度不一致")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("參數包含 NaN 或 Inf")
    layers.append((weight, bias))

if not layers:
    raise ValueError("模型中找不到 Gemm 全連接層")

print(layers[0][0])

output_dir = base_dir / "output/parameters"
output_dir.mkdir(parents=True, exist_ok=True)
weight_path = output_dir / "weighted_matrix.dat"
bias_path = output_dir / "bias.dat"

# 輸入層為第 1 層，節點從 1 開始；後續各層接續編號。
source_offset = 0
with weight_path.open("w") as weight_file, bias_path.open("w") as bias_file:
    bias_file.write("# global_node  layer_id  local_node  bias\n")
    for layer_id, (weight, bias) in enumerate(layers, start=2):
        n_out, n_in = weight.shape
        target_offset = source_offset + n_in
        for target in range(n_out):
            global_target = target_offset + target + 1
            for source in range(n_in):
                global_source = source_offset + source + 1
                weight_file.write(
                    f"{global_target} {global_source} {weight[target, source]:.16e}\n"
                )
            bias_file.write(
                f"{global_target} {layer_id} {target + 1} {bias[target]:.16e}\n"
            )
        source_offset = target_offset

layer_sizes = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
print("網路架構：", " → ".join(map(str, layer_sizes)))
print(f"權重：{weight_path}（{sum(w.size for w, _ in layers)} 筆）")
print(f"Bias：{bias_path}（{sum(b.size for _, b in layers)} 筆）")
