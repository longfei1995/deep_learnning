import torch 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        
        self.down1 = ConvBlock(input_channels, 64)
        self.down2 = ConvBlock(64, 128)
        
        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128 + 256, 128)
        self.up1 = ConvBlock(64 + 128, 64)
        self.out = nn.Conv2d(64, input_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        
    def forward(self, x):
        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        x = self.bot1(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = self.out(x)
        return x
        
if __name__ == "__main__":
    model = UNet(input_channels=1)
    x = torch.randn(10, 1, 28, 28)  # batch_size=10, channels=1, height=28, width=28
    output = model(x)
    print(output.shape)  # 应该输出torch.Size([10, 1, 28, 28])