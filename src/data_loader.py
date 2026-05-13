"""
CIFAR-10 数据加载与预处理模块
支持从本地已下载的数据集加载，兼容跨平台路径
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10LocalDataset(Dataset):
    """CIFAR-10 本地数据集类。

    从本地 cifar-10-batches-py 目录加载数据，支持训练集和测试集加载，
    并提供可选的图像变换操作。
    """

    def __init__(self, data_path: str, train: bool = True, transform: Optional[transforms.Compose] = None) -> None:
        """初始化数据集。

        Args:
            data_path: CIFAR-10 数据目录路径。
            train: 是否为训练集（若为 False 则加载测试集）。
            transform: 可选的图像变换操作。
        """
        self.data_path = data_path
        self.train = train
        self.transform = transform

        if self.train:
            # 加载 5 个训练批次
            self.images, self.labels = self._load_batches(
                [f"data_batch_{i}" for i in range(1, 6)]
            )
        else:
            # 加载测试批次
            self.images, self.labels = self._load_batches(["test_batch"])

    def _load_batches(self, batch_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """加载指定的 CIFAR-10 batch 文件。

        Args:
            batch_names: 要加载的 batch 文件名列表。

        Returns:
            (images, labels) 元组，images 形状为 (N, 3, 32, 32)，
            labels 形状为 (N,)。
        """
        images = []
        labels = []
        for name in batch_names:
            filepath = os.path.join(self.data_path, name)
            with open(filepath, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            # 图像数据: (10000, 3072) -> (10000, 3, 32, 32)
            batch_images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            images.append(batch_images)
            labels.extend(batch[b'labels'])
        images = np.concatenate(images, axis=0)
        labels = np.array(labels, dtype=np.int64)
        return images, labels

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取指定索引的样本。

        Args:
            idx: 样本索引。

        Returns:
            (image, label) 元组，image 为经过变换后的图像张量，label 为类别标签。
        """
        image = self.images[idx]  # (3, 32, 32)
        label = self.labels[idx]

        # 转换为 PIL Image 以便使用 torchvision transforms
        from PIL import Image
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))  # CHW -> HWC

        if self.transform:
            image = self.transform(image)

        return image, label


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ])


def get_ablation_transforms(aug_type: str) -> transforms.Compose:
    if aug_type == 'no_aug':
        return transforms.Compose([transforms.ToTensor()])
    elif aug_type == 'normalize_only':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ])
    elif aug_type == 'full_aug':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ])
    else:
        raise ValueError(f"Unknown aug_type: {aug_type}")


def get_data_loaders(
    data_path: str,
    batch_size: int = 128,
    num_workers: int = 4,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if train_transform is None:
        train_transform = get_transforms(train=True)
    if val_transform is None:
        val_transform = get_transforms(train=False)
    test_transform = val_transform

    # 完整训练集（后续手动划分 val）
    full_train_dataset = CIFAR10LocalDataset(
        data_path=data_path,
        train=True,
        transform=train_transform
    )

    # 分层抽样: 每类别取 10% 作为验证集，确保类别均衡
    from torch.utils.data import Subset
    labels: np.ndarray = full_train_dataset.labels
    rng = np.random.RandomState(42)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for c in range(10):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)
        n_val = len(class_indices) // 10  # 每类 500 张
        val_indices.extend(class_indices[:n_val].tolist())
        train_indices.extend(class_indices[n_val:].tolist())

    train_dataset = Subset(full_train_dataset, train_indices)

    val_full_dataset = CIFAR10LocalDataset(
        data_path=data_path, train=True, transform=val_transform
    )
    val_dataset = Subset(val_full_dataset, val_indices)

    test_dataset = CIFAR10LocalDataset(
        data_path=data_path,
        train=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


if __name__ == "__main__":
    # 简单测试
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "cifar-10-batches-py")

    train_loader, val_loader, test_loader = get_data_loaders(data_path, batch_size=64)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break
