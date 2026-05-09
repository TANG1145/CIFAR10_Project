"""
卷积神经网络 (CNN) 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    卷积神经网络
    结构: Conv -> BN -> ReLU -> Pool -> ... -> FC
    """

    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(CNN, self).__init__()

        # 卷积块 1: 3 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            nn.Dropout2d(dropout_rate / 2)
        )

        # 卷积块 2: 64 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            nn.Dropout2d(dropout_rate / 2)
        )

        # 卷积块 3: 128 -> 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
            nn.Dropout2d(dropout_rate / 2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


class ImprovedCNN(nn.Module):
    """
    改进版 CNN：更深的网络 + 残差连接思想
    更容易达到 85%+ 准确率
    """

    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(ImprovedCNN, self).__init__()

        self.conv1 = self._make_layer(3, 64, 2, dropout_rate / 2)
        self.conv2 = self._make_layer(64, 128, 2, dropout_rate / 2)
        self.conv3 = self._make_layer(128, 256, 2, dropout_rate / 2)
        self.conv4 = self._make_layer(256, 512, 2, dropout_rate / 2)

        # 全局平均池化替代全连接
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, dropout):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2 if in_channels != out_channels else 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)  # 32
        x = self.conv2(x)  # 16
        x = self.conv3(x)  # 8
        x = self.conv4(x)  # 4
        x = self.global_pool(x)  # 1x1
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_cnn(variant='standard', **kwargs):
    """工厂函数"""
    if variant == 'improved':
        return ImprovedCNN(**kwargs)
    return CNN(**kwargs)


if __name__ == "__main__":
    for name, model in [("Standard CNN", CNN()), ("Improved CNN", ImprovedCNN())]:
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: Output shape {out.shape}, Parameters: {total_params:,}")
