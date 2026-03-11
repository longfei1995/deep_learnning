import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



if __name__== "__main__":
    # 测试sigmoid函数 和 relu函数
    x = torch.linspace(-10, 10, 1000).unsqueeze(1)
    sigmoid_x = F.sigmoid(x)
    relu_x = F.relu(x)
    
    # 绘图
    fig = plt.figure(figsize=(8, 6))
    fig1 = fig.add_subplot(1, 2, 1)    
    fig1.plot(x.numpy(), sigmoid_x.numpy(), label='Sigmoid')
    
    fig2 = fig.add_subplot(1, 2, 2)
    fig2.plot(x.numpy(), relu_x.numpy(), label='ReLU', color='orange')
    
    fig1.legend()
    fig2.legend()
    plt.show()