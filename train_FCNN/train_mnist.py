from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
import onnx
import onnxruntime as ort

torch.manual_seed(1) # Set random seed for reproducibility

output_dir = Path("output") # Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# 1. 讀取資料
# ToTensor 將像素從 0～255 轉為 0～1 的浮點數。
transform = transforms.ToTensor()
# 2. 下載 MNIST 資料集
train_data = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
# 3. 下載 MNIST 測試資料集
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)
# 4. 將訓練資料集拆分為訓練集與驗證集
train_subset, val_subset = random_split(
    train_data,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(
    train_subset, batch_size=128, shuffle=True
)
val_loader = DataLoader(
    val_subset, batch_size=256, shuffle=False
)
# 5. 建立 DataLoader 物件，將測試資料集分批次讀取
test_loader = DataLoader(
    test_data, batch_size=256, shuffle=False
)

# 6. 建立神經網路模型
class FCNN(nn.Module):
    def __init__(self): # 定義神經網路的結構
        super().__init__()
        self.flatten = nn.Flatten() # 將 28x28 的影像展平為 784 維的向量
        self.fc1 = nn.Linear(784, 100) # 全連接層，輸入 784 維，輸出 256 維
        # self.fc2 = nn.Linear(256, 256) # 全連接層，輸入 256 維，輸出 256 維
        self.fc2 = nn.Linear(100, 10) # 全連接層，輸入 256 維，輸出 10 維 (對應 10 個類別)

    def forward(self, x): # 定義前向傳播函數
        x = self.flatten(x) # 將影像展平為向量
        x = torch.relu(self.fc1(x)) # 使用 ReLU 激活函數
        # x = torch.relu(self.fc2(x))
        return self.fc2(x)

# 7. 建立模型、損失函數和優化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")
model = FCNN().to(device)
loss_fn = nn.CrossEntropyLoss()

lr_max = 0.01
epochs_per_cycle = 45
num_cycles = 20
epochs = epochs_per_cycle * num_cycles
optimizer = torch.optim.SGD(model.parameters(), lr=lr_max)

# 8. 訓練模型
for epoch in range(epochs):
    cycle_id = epoch // epochs_per_cycle
    epoch_in_cycle = epoch % epochs_per_cycle
    # epoch 從 0 開始；每個 epoch 固定 LR，cycle 邊界只重設 LR。
    # 餘數為 0..44，因此最後一輪接近零但不取零。
    lr = 0.5 * lr_max * (
        1.0 + np.cos(np.pi * epoch_in_cycle / epochs_per_cycle)
    )
    for group in optimizer.param_groups:
        group["lr"] = lr

    model.train()
    train_loss_sum = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # 清空梯度

        scores = model(images)
        loss = loss_fn(scores, labels) # 計算損失

        loss.backward() # 反向傳播計算梯度
        optimizer.step() # 更新模型參數

        train_loss_sum += loss.item() * images.size(0) # 累加損失，乘以批次大小以獲得總損失

    train_loss = train_loss_sum / len(train_loader.dataset)

    # 每輪評估驗證集，不計算梯度，也不更新模型參數。
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            loss = loss_fn(scores, labels)
            val_loss_sum += loss.item() * images.size(0)
            val_correct += (scores.argmax(dim=1) == labels).sum().item()

    val_loss = val_loss_sum / len(val_loader.dataset)
    val_accuracy = val_correct / len(val_loader.dataset)

    print(
        f"Cycle {cycle_id + 1}/{num_cycles} | "
        f"Epoch {epoch_in_cycle + 1}/{epochs_per_cycle} "
        f"(global {epoch + 1}/{epochs}) | "
        f"lr={lr:.6e} | "
        f"train loss={train_loss:.4f} | "
        f"val loss={val_loss:.4f} | "
        f"val accuracy={val_accuracy:.2%}"
    )

# SGD protocol 只保存訓練結束時的參數；validation 僅作監測。
# 測試與 ONNX 匯出沿用這組最終權重。
torch.save(model.state_dict(), output_dir / "final_fcnn.pth")
print(f"儲存最終模型：Epoch {epochs}, val loss={val_loss:.4f}")

# 將模型設為評估模式，這樣在測試時會關閉 dropout 和 batch normalization 等層的訓練行為
model.eval() 
correct = 0
total = 0

with torch.no_grad(): # 在測試時不需要計算梯度，節省記憶體和計算資源
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        predictions = scores.argmax(dim=1) # 取得每個樣本的預測類別
        correct += (predictions == labels).sum().item() # 累加正確預測的數量
        total += labels.size(0) # 累加總樣本數量

print(f"Test accuracy = {100 * correct / total:.2f}%")

# 匯出與 ONNX Runtime 比對都在 CPU 上進行。
model = model.cpu()
example_input = torch.zeros(1, 1, 28, 28)
torch.onnx.export(
    model,
    (example_input,),
    str(output_dir / "mnist_fcnn.onnx"),
    input_names=["image"],
    output_names=["scores"],
    dynamo=True,
    external_data=False,
)

# 9. 驗證 ONNX 模型
onnx_model = onnx.load(str(output_dir / "mnist_fcnn.onnx"))
onnx.checker.check_model(onnx_model)

image, label = test_data[0]
image = image.unsqueeze(0)  # [1, 1, 28, 28]

with torch.no_grad():
    pytorch_scores = model(image).numpy()

session = ort.InferenceSession(
    str(output_dir / "mnist_fcnn.onnx"),
    providers=["CPUExecutionProvider"],
)
onnx_scores = session.run(None, {"image": image.numpy()})[0]

np.testing.assert_allclose(
    pytorch_scores, onnx_scores, rtol=1e-4, atol=1e-5
)

print("ONNX 與 PyTorch 輸出比對通過")
print("正確答案：", label)
print("ONNX 預測：", int(onnx_scores.argmax(axis=1)[0]))
