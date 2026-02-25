"""
使用最佳模型进行测试
"""

import torch
from model import alexnet
from utils.datasets import load_datasets

# 加载测试数据
_, test_dataloader = load_datasets("cifar100", "~/pytorch_datasets", 256)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = alexnet()

# 加载最佳模型权重
checkpoint = torch.load("./checkpoints/cifar100_best.pth", map_location=device)
state_dict = checkpoint["state_dict"]

# 处理DataParallel保存的模型（权重key有'module.'前缀）
# 如果checkpoint是用DataParallel保存的，需要去掉'module.'前缀
if list(state_dict.keys())[0].startswith("module."):
    print("检测到DataParallel模型，正在转换...")
    # 方法1：去掉'module.'前缀
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 去掉'module.'前缀 (7个字符)
        new_state_dict[name] = v
    state_dict = new_state_dict

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("=" * 60)
print("测试最佳模型")
print("=" * 60)
print(f"模型来自 Epoch: {checkpoint['epoch']}")
print(f"训练时的最佳准确率: {checkpoint['best_prec1']:.2f}%")
print(f"设备: {device}")
print("=" * 60)

# 测试
correct = 0
total = 0

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        if (i + 1) % 10 == 0:
            print(f"已测试 {total}/10000 张图像...", end="\r")

accuracy = 100 * correct / total
print(f"\n\n🎯 测试集准确率: {accuracy:.2f}%")
print(f"正确分类: {correct}/{total}")
print("=" * 60)
