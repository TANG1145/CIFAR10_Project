"""
模型模块初始化
导出所有模型类和工厂函数
"""

from typing import Any, Dict

import torch.nn as nn

from models.mlp import MLP, create_mlp
from models.cnn import CNN, create_cnn
from models.resnet import ResNet, BasicBlock, create_resnet18


def create_model(model_type: str, **kwargs: Any) -> nn.Module:
    if model_type == 'mlp':
        return create_mlp(dropout_rate=kwargs.get('dropout_rate', 0.3))
    elif model_type == 'cnn':
        return create_cnn(
            variant=kwargs.get('variant', 'improved'),
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            use_bn=kwargs.get('use_bn', True)
        )
    elif model_type == 'resnet18':
        return create_resnet18()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
