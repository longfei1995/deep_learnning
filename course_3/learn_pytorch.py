"""
PyTorch 基础教程
适合新手的 PyTorch 使用示例代码
PyTorch 版本: 2.9.1+cpu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=" * 60)
print("PyTorch 基础教程")
print("=" * 60)

# ========================================
# 1. 张量（Tensor）的创建
# ========================================
print("\n【1. 张量的创建】")

# 从列表创建张量
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"从列表创建: {tensor_from_list}")

# 创建二维张量（矩阵）
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(f"二维张量:\n{matrix}")

# 创建全零张量
zeros = torch.zeros(3, 4)  # 3行4列
print(f"全零张量:\n{zeros}")

# 创建全一张量
ones = torch.ones(2, 3)
print(f"全一张量:\n{ones}")

# 创建随机张量（0-1之间的均匀分布）
random_tensor = torch.rand(2, 3)
print(f"随机张量:\n{random_tensor}")

# 创建标准正态分布的随机张量
randn_tensor = torch.randn(2, 3)
print(f"正态分布随机张量:\n{randn_tensor}")

# 创建指定范围的张量
arange_tensor = torch.arange(0, 10, 2)  # 从0到10，步长为2
print(f"指定范围张量: {arange_tensor}")

# 创建等间隔张量
linspace_tensor = torch.linspace(0, 1, 5)  # 从0到1，分成5个点
print(f"等间隔张量: {linspace_tensor}")


# ========================================
# 2. 张量的基本属性
# ========================================
print("\n【2. 张量的基本属性】")

x = torch.randn(3, 4, 5)
print(f"张量形状: {x.shape}")  # 或者 x.size()
print(f"张量维度: {x.ndim}")
print(f"张量元素总数: {x.numel()}")
print(f"张量数据类型: {x.dtype}")
print(f"张量所在设备: {x.device}")


# ========================================
# 3. 张量的形状操作
# ========================================
print("\n【3. 张量的形状操作】")

# reshape：改变形状
a = torch.arange(12)
print(f"原始张量: {a}")
b = a.reshape(3, 4)
print(f"reshape成3x4:\n{b}")

# view：类似reshape，但要求内存连续
c = b.view(2, 6)
print(f"view成2x6:\n{c}")

# 转置
d = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(f"原始矩阵:\n{d}")
print(f"转置后:\n{d.T}")

# squeeze：移除大小为1的维度
e = torch.zeros(1, 3, 1, 4)
print(f"squeeze前的形状: {e.shape}")
e_squeezed = e.squeeze()
print(f"squeeze后的形状: {e_squeezed.shape}")

# unsqueeze：增加一个大小为1的维度
f = torch.tensor([1, 2, 3])
print(f"unsqueeze前的形状: {f.shape}")
f_unsqueezed = f.unsqueeze(0)  # 在第0维增加
print(f"unsqueeze后的形状: {f_unsqueezed.shape}")
print(f"结果:\n{f_unsqueezed}")


# ========================================
# 4. 张量的数学运算
# ========================================
print("\n【4. 张量的数学运算】")

# 基本运算
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])

print(f"加法: {x + y}")
print(f"减法: {x - y}")
print(f"乘法（逐元素）: {x * y}")
print(f"除法: {x / y}")
print(f"幂运算: {x ** 2}")

# 矩阵乘法
A = torch.tensor([[1., 2.],
                  [3., 4.]])
B = torch.tensor([[5., 6.],
                  [7., 8.]])

print(f"矩阵A:\n{A}")
print(f"矩阵B:\n{B}")
print(f"矩阵乘法 A @ B:\n{A @ B}")  # 或者 torch.matmul(A, B)

# 统计运算
data = torch.tensor([[1., 2., 3.],
                     [4., 5., 6.]])
print(f"数据:\n{data}")
print(f"求和: {data.sum()}")
print(f"按行求和: {data.sum(dim=0)}")  # 沿着第0维（行）
print(f"按列求和: {data.sum(dim=1)}")  # 沿着第1维（列）
print(f"平均值: {data.mean()}")
print(f"最大值: {data.max()}")
print(f"最小值: {data.min()}")


# ========================================
# 5. 张量与NumPy的转换
# ========================================
print("\n【5. 张量与NumPy的转换】")

# PyTorch张量 -> NumPy数组
torch_tensor = torch.tensor([1, 2, 3, 4, 5])
numpy_array = torch_tensor.numpy()
print(f"PyTorch张量: {torch_tensor}")
print(f"转换为NumPy: {numpy_array}")
print(f"NumPy数组类型: {type(numpy_array)}")

# NumPy数组 -> PyTorch张量
np_array = np.array([6, 7, 8, 9, 10])
torch_from_numpy = torch.from_numpy(np_array)
print(f"NumPy数组: {np_array}")
print(f"转换为PyTorch: {torch_from_numpy}")


# ========================================
# 6. 自动求导（Autograd）
# ========================================
print("\n【6. 自动求导（Autograd）】")

# 需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")

# 定义函数 y = x^2 + 3x + 1
y = x ** 2 + 3 * x + 1
print(f"y = x^2 + 3x + 1 = {y}")

# 反向传播计算梯度
y.backward()

# dy/dx = 2x + 3，当x=2时，梯度应该是7
print(f"dy/dx (x={x.item()}) = {x.grad}")

# 多变量求导示例
x1 = torch.tensor([1.0], requires_grad=True)
x2 = torch.tensor([2.0], requires_grad=True)
z = x1 ** 2 + x2 ** 3
print(f"\nz = x1^2 + x2^3 = {z}")

z.backward()
print(f"dz/dx1 = 2*x1 = {x1.grad}")  # 应该是2
print(f"dz/dx2 = 3*x2^2 = {x2.grad}")  # 应该是12


# ========================================
# 7. 简单的线性回归示例
# ========================================
print("\n【7. 简单的线性回归示例】")

# 生成训练数据：y = 2x + 3 + noise
torch.manual_seed(42)  # 设置随机种子以便结果可复现
X_train = torch.randn(100, 1)  # 100个样本，1个特征
y_train = 2 * X_train + 3 + 0.1 * torch.randn(100, 1)  # 添加噪声

# 定义线性模型
w = torch.randn(1, 1, requires_grad=True)  # 权重
b = torch.randn(1, requires_grad=True)     # 偏置

# 学习率
learning_rate = 0.01

# 训练
print("开始训练...")
for epoch in range(100):
    # 前向传播
    y_pred = X_train @ w + b
    
    # 计算损失（均方误差）
    loss = ((y_pred - y_train) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 更新参数（梯度下降）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

print(f"训练完成！")
print(f"学到的权重 w: {w.item():.4f} (真实值: 2.0)")
print(f"学到的偏置 b: {b.item():.4f} (真实值: 3.0)")


# ========================================
# 8. 使用nn.Module定义神经网络
# ========================================
print("\n【8. 使用nn.Module定义神经网络】")

# 定义一个简单的两层全连接网络
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层
        self.relu = nn.ReLU()                          # 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size) # 第二层
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建网络实例
model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
print(f"网络结构:\n{model}")

# 查看网络参数
print("\n网络参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 前向传播测试
test_input = torch.randn(5, 10)  # 批量大小为5，输入维度为10
output = model(test_input)
print(f"\n输入形状: {test_input.shape}")
print(f"输出形状: {output.shape}")


# ========================================
# 9. 完整的训练流程示例（分类问题）
# ========================================
print("\n【9. 完整的训练流程示例】")

# 生成模拟的二分类数据
torch.manual_seed(42)
n_samples = 200
X = torch.randn(n_samples, 2)  # 2个特征
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)  # 标签

# 定义模型
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = BinaryClassifier()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

# 训练循环
num_epochs = 50
batch_size = 32

print("开始训练分类器...")
for epoch in range(num_epochs):
    # 打乱数据
    indices = torch.randperm(n_samples)
    
    epoch_loss = 0.0
    for i in range(0, n_samples, batch_size):
        # 获取批次数据
        batch_indices = indices[i:i+batch_size]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / (n_samples // batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 评估模型
model.eval()  # 设置为评估模式
with torch.no_grad():
    predictions = model(X)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y).float().mean()
    print(f"\n训练集准确率: {accuracy.item() * 100:.2f}%")


# ========================================
# 10. 保存和加载模型
# ========================================
print("\n【10. 保存和加载模型】")

# 保存模型
model_path = "simple_model.pth"
torch.save(model.state_dict(), model_path)
print(f"模型已保存到: {model_path}")

# 加载模型
loaded_model = BinaryClassifier()
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
print("模型已加载")

# 验证加载的模型
with torch.no_grad():
    test_input = torch.randn(1, 2)
    original_output = model(test_input)
    loaded_output = loaded_model(test_input)
    print(f"原始模型输出: {original_output.item():.4f}")
    print(f"加载模型输出: {loaded_output.item():.4f}")
    print(f"输出是否一致: {torch.allclose(original_output, loaded_output)}")


# ========================================
# 11. 常用技巧总结
# ========================================
print("\n【11. 常用技巧总结】")

print("""
PyTorch 常用技巧：

1. 设置随机种子以便结果可复现：
   torch.manual_seed(42)

2. 检查是否有GPU可用：
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

3. 将模型和数据移动到GPU：
   model.to(device)
   data = data.to(device)

4. 保存完整模型（包括结构）：
   torch.save(model, 'model_complete.pth')

5. 只保存模型参数（推荐）：
   torch.save(model.state_dict(), 'model_weights.pth')

6. 训练模式和评估模式：
   model.train()  # 训练模式（启用dropout、batch norm等）
   model.eval()   # 评估模式（禁用dropout、batch norm等）

7. 不计算梯度（推理时节省内存）：
   with torch.no_grad():
       output = model(input)

8. 查看张量的设备位置：
   print(tensor.device)

9. 克隆张量（深拷贝）：
   new_tensor = old_tensor.clone()

10. 分离计算图（不计算梯度）：
    detached_tensor = tensor.detach()
""")

print("\n" + "=" * 60)
print("教程完成！继续探索PyTorch的更多功能吧！")
print("=" * 60)

