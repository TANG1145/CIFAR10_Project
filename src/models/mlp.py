"""
多层感知机 (MLP) 模型
作为图像分类的基线模型
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    多层感知机
    输入: 32x32x3 = 3072 展平向量
    输出: 10 类 logits
    """

    def __init__(self, input_dim=3072, hidden_dims=[1024, 512, 256],
                 num_classes=10, dropout_rate=0.3):
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, 32, 32) -> flatten -> (B, 3072)
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


def create_mlp(**kwargs):
    """工厂函数，方便创建不同配置的 MLP"""
    return MLP(**kwargs)


if __name__ == "__main__":
    model = create_mlp()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
