"""
检查点管理模块

提供模型检查点的保存与加载功能。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    path: str,
) -> None:
    """保存模型检查点。

    Args:
        model: 要保存的模型。
        optimizer: 优化器（保存其状态字典）。
        epoch: 当前训练轮数。
        best_acc: 最佳验证准确率。
        path: 保存路径。
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device,
) -> Tuple[int, float]:
    """加载模型检查点。

    Args:
        model: 要加载权重的模型。
        optimizer: 优化器（若为 None 则跳过加载优化器状态）。
        path: 检查点路径。
        device: 计算设备映射。

    Returns:
        (epoch, best_acc) 元组，分别表示上次保存的轮数和最佳准确率。
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)
