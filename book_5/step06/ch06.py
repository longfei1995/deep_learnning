import torch
import torch.nn as nn
import torch.nn.functional as F
class PredictionModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        y = self.linear1(x)
        y = F.sigmoid(y)  # 激活函数    
        y = self.linear2(y)
        return y





if __name__== "__main__":
    # 数据准备
    torch.manual_seed(0)
    x = torch.rand(100, 1)
    y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)   # 添加一些噪声
    
    # 超参数
    lr = 0.2
    iters = int(1e4)
    
    # 训练
    model = PredictionModel()
    # 优化器生成
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(iters):
        # 前向传播
        y_pred = model(x)
        # 计算损失
        loss = nn.MSELoss()(y_pred, y)
        # 反向传播
        loss.backward()
        optimizer.step()    # 参数更新
        optimizer.zero_grad()   # 梯度清零
        # 打印损失
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{iters}], Loss: {loss.item():.4f}')
        
    
    # 绘图,展示结果
    import matplotlib.pyplot as plt
    plt.scatter(x.numpy(), y.numpy(), label='Data')
    x_line = torch.linspace(0, 1, 100).unsqueeze(1)
    y_line = model(x_line).detach().numpy()
    plt.plot(x_line.numpy(), y_line, color='red', label='Fitted Line')
    plt.legend()
    plt.show() 