"""
多层感知机 (MLP) 模型
作为图像分类的基线模型
"""

from typing import List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机。

    输入为 32x32x3 = 3072 维展平向量，输出为 10 类 logits。
    包含多个全连接层，每层后接 BatchNorm、ReLU 和 Dropout。
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: List[int] = [1024, 512, 256],
        num_classes: int = 10,
        dropout_rate: float = 0.3,
    ) -> None:
        """初始化 MLP。

        Args:
            input_dim: 输入特征维度。
            hidden_dims: 隐藏层维度列表。
            num_classes: 分类数量。
            dropout_rate: Dropout 率。
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化网络权重（He 初始化）。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 (B, 3, 32, 32)。

        Returns:
            输出 logits，形状为 (B, 10)。
        """
        x = x.view(x.size(0), -1)  # (B, 3, 32, 32) -> (B, 3072)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


def create_mlp(**kwargs: dict) -> MLP:
    """创建 MLP 模型的工厂函数。

    Args:
        **kwargs: 传递给 MLP 构造函数的参数（input_dim, hidden_dims, num_classes, dropout_rate）。

    Returns:
        MLP 模型实例。
    """
    return MLP(**kwargs)


if __name__ == "__main__":
    model = create_mlp()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
