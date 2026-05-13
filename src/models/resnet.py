"""
ResNet-18 for CIFAR-10
参考: https://github.com/kuangliu/pytorch-cifar
"""

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet 基本残差块。

    包含两个 3x3 卷积层和跳跃连接，当 stride != 1 或维度不匹配时使用 1x1 卷积调整。
    """

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """初始化 BasicBlock。

        Args:
            in_planes: 输入通道数。
            planes: 输出通道数。
            stride: 卷积步长。
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量。

        Returns:
            经过残差块处理后的输出张量。
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet 网络。

    由多个残差层组成，每层包含若干 BasicBlock，支持自定义 block 类型和层数配置。
    """

    def __init__(
        self,
        block: Type[BasicBlock],
        num_blocks: List[int],
        num_classes: int = 10,
    ) -> None:
        """初始化 ResNet。

        Args:
            block: 残差块类型（如 BasicBlock）。
            num_blocks: 每层残差块数量列表。
            num_classes: 分类数量。
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: Type[BasicBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """构建一个残差层。

        Args:
            block: 残差块类型。
            planes: 输出通道数。
            num_blocks: 残差块数量。
            stride: 第一个残差块的卷积步长。

        Returns:
            由多个残差块组成的 Sequential 模块。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 (B, 3, 32, 32)。

        Returns:
            输出 logits，形状为 (B, num_classes)。
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_resnet18(num_classes: int = 10) -> ResNet:
    """创建 ResNet-18 模型的工厂函数。

    Args:
        num_classes: 分类数量。

    Returns:
        ResNet-18 模型实例。
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


if __name__ == "__main__":
    model = create_resnet18()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-18: Output shape {out.shape}, Parameters: {total_params:,}")
