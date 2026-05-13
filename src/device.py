"""
设备管理模块
统一的设备检测和选择逻辑
"""

import torch


def get_device(device_arg: str = 'auto') -> torch.device:
    """获取计算设备

    根据参数自动选择或指定计算设备，支持跨平台（Mac CPU / Win CUDA）。

    Args:
        device_arg: 设备参数，'auto' 自动选择（有CUDA就用GPU否则CPU），
                    或 'cuda', 'cpu' 等 torch 支持的设备名

    Returns:
        torch.device: 选择的计算设备
    """
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    return device
