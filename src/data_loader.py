"""
CIFAR-10 数据加载与预处理模块
支持从本地已下载的数据集加载，兼容跨平台路径
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10LocalDataset(Dataset):
    """从本地 cifar-10-batches-py 目录加载数据"""

    def __init__(self, data_path, train=True, transform=None):
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

    def _load_batches(self, batch_names):
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # (3, 32, 32)
        label = self.labels[idx]

        # 转换为 PIL Image 以便使用 torchvision transforms
        from PIL import Image
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))  # CHW -> HWC

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train=True):
    """获取数据预处理变换"""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616])
        ])


def get_data_loaders(data_path, batch_size=128, num_workers=4):
    """
    创建训练、验证、测试 DataLoader
    将训练集最后 5000 张作为验证集
    """
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    # 完整训练集（后续手动划分 val）
    full_train_dataset = CIFAR10LocalDataset(
        data_path=data_path,
        train=True,
        transform=train_transform
    )

    # 划分训练/验证
    train_size = len(full_train_dataset) - 5000
    val_size = 5000
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 验证集使用测试变换（无数据增强）
    val_dataset.dataset.transform = test_transform

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
