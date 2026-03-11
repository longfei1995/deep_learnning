import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets


# 超参数的设置
input_dim = 784 # 图像数据大小28*28
hidden_dim = 200 # 隐藏层神经元数量
latent_dim = 20 # 潜在变量z的维度
epochs = 30
lr = 3e-4
batch_size = 32

# 定义Encoder网络
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 对数方差

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        sigma = torch.exp(0.5 * logvar)  # 标准差
        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))  # 输出在0-1之间
        return x_recon
    
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)  # 从标准正态分布采样
        z = mu + eps * sigma  # 重参数化技巧
        return z

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_recon = self.decoder(z)
        return x_recon, mu, sigma
    
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction='sum') # 重构误差
        L2 = torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) # KL散度
        loss = (L1 - L2) / batch_size
        return loss
    
    
if __name__ == "__main__":
    # 进行训练
    # 1. 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten) # 将28*28的图像展平为784维的向量
        ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. 模型实例化和优化器设置
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    # 3. 训练循环
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        for x, label in train_loader:
            optimizer.zero_grad()   # 清空梯度
            loss = model.get_loss(x)    # 计算损失
            loss.backward()             # 反向传播计算梯度
            optimizer.step()            # 更新参数
            loss_sum += loss.item()     # 累加损失
            cnt += 1
        avg_loss = loss_sum / cnt
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    # 4. 生成新图像
    with torch.no_grad():
        sample_size = 64
        z = torch.randn(sample_size, latent_dim)  # 从标准正态分布采样潜在变量
        generated_iamges = model.decoder(z).view(sample_size, 1, 28, 28)  # 将生成的图像重塑为28*28的格式
        
        grid_img = torchvision.utils.make_grid(generated_iamges, nrow=8, padding=2, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()