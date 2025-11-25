import torch
import torch.nn as nn

__all__ = ["AlexNet", "alexnet"]


class AlexNet(nn.Module):

    def __init__(self, classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # layer 1 - 适配32x32输入：减小kernel和stride
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            # layer 2
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            # layer 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # layer 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # layer 5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            # fc 1 - 调整输入维度：256 * 4 * 4 = 4096
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 2048),  # 减小全连接层大小以适配小图像
            nn.ReLU(inplace=True),
            # fc 2
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # fc 3
            nn.Linear(2048, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
