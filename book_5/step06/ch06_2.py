import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    transform = transforms.ToTensor()  # 将图像转换为张量

    # 读取MNIST数据集
    mnist_train = torchvision.datasets.MNIST(
        root="./data", 
        transform=transform,    # 指定预处理函数
        train=True, 
        download=True
    )
    
    # 将数据做成小批量形式
    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_train,
        batch_size=32,
        shuffle=True
    )
    
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break  # 只查看第一个批次的数据
    