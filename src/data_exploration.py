"""
CIFAR-10 数据探索可视化脚本
生成类别分布、样本展示、像素统计、数据增强对比等可视化图表
"""

import os
import sys
import argparse
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

for font_name in ['Arial Unicode MS', 'Hiragino Sans GB', 'PingFang HK', 'Hei']:
    try:
        matplotlib.rcParams['font.family'] = font_name
        matplotlib.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

from PIL import Image
import torch
from torchvision import transforms

# 复用 data_loader.py 中的类和函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import CIFAR10_CLASSES, get_transforms

# CIFAR-10 中文标签（可选）
CIFAR10_CLASSES_CN = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]

# Normalize 参数（用于对比和反归一化）
NORMALIZE_MEAN = np.array([0.4914, 0.4822, 0.4465])
NORMALIZE_STD = np.array([0.2470, 0.2435, 0.2616])


def load_batches(
    data_path: str, batch_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """加载指定 batch 文件，返回图像和标签。

    Args:
        data_path: CIFAR-10 数据目录路径。
        batch_names: 要加载的 batch 文件名列表。

    Returns:
        (images, labels) 元组，images 形状为 (N, 3, 32, 32)，
        labels 形状为 (N,)。
    """
    images = []
    labels = []
    for name in batch_names:
        filepath = os.path.join(data_path, name)
        with open(filepath, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        batch_images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        images.append(batch_images)
        labels.extend(batch[b'labels'])
    images = np.concatenate(images, axis=0)
    labels = np.array(labels, dtype=np.int64)
    return images, labels


def plot_class_distribution(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    output_dir: str,
) -> None:
    """绘制类别分布条形图。

    展示训练集、验证集、测试集中每个类别的样本数量分布。

    Args:
        train_labels: 训练集标签数组。
        val_labels: 验证集标签数组。
        test_labels: 测试集标签数组。
        output_dir: 图片输出目录。
    """
    num_classes = len(CIFAR10_CLASSES)
    train_counts = np.bincount(train_labels, minlength=num_classes)
    val_counts = np.bincount(val_labels, minlength=num_classes)
    test_counts = np.bincount(test_labels, minlength=num_classes)

    x = np.arange(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, train_counts, width, label='训练集 (Train)', color='steelblue')
    bars2 = ax.bar(x, val_counts, width, label='验证集 (Val)', color='coral')
    bars3 = ax.bar(x + width, test_counts, width, label='测试集 (Test)', color='seagreen')

    ax.set_xlabel('类别')
    ax.set_ylabel('样本数量')
    ax.set_title('CIFAR-10 各类别样本数量分布')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cn}\n({en})" for cn, en in zip(CIFAR10_CLASSES_CN, CIFAR10_CLASSES)],
                       fontsize=9)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 在条形图上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[已保存] 类别分布图: {save_path}")


def plot_sample_images(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    output_dir: str,
    samples_per_class: int = 8,
) -> None:
    """绘制每个类别的随机样本展示图（2x4 网格）。

    Args:
        train_images: 训练集图像数组，形状为 (N, 3, 32, 32)。
        train_labels: 训练集标签数组。
        output_dir: 图片输出目录。
        samples_per_class: 每类展示的样本数量。
    """
    num_classes = len(CIFAR10_CLASSES)
    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(samples_per_class * 1.5, num_classes * 1.5))

    for cls in range(num_classes):
        indices = np.where(train_labels == cls)[0]
        selected = np.random.choice(indices, samples_per_class, replace=False)
        for i, idx in enumerate(selected):
            row = cls
            col = i
            img = train_images[idx]  # (3, 32, 32)
            img = np.transpose(img, (1, 2, 0))
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if i == 0:
                axes[row, col].set_title(
                    f"{CIFAR10_CLASSES_CN[cls]} ({CIFAR10_CLASSES[cls]})",
                    fontsize=9, loc='left'
                )

    plt.suptitle('CIFAR-10 每类随机样本展示 (8张)', fontsize=14, y=1.00)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_images.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[已保存] 样本图片展示: {save_path}")


def plot_pixel_statistics(train_images: np.ndarray, output_dir: str) -> None:
    """绘制像素统计柱状图。

    展示每个通道（R/G/B）的像素均值与标准差，并与数据归一化参数进行对比。

    Args:
        train_images: 训练集图像数组，形状为 (N, 3, 32, 32)。
        output_dir: 图片输出目录。
    """
    # train_images shape: (N, 3, 32, 32)
    mean_computed = train_images.mean(axis=(0, 2, 3))
    std_computed = train_images.std(axis=(0, 2, 3))

    channels = ['R (红)', 'G (绿)', 'B (蓝)']
    x = np.arange(len(channels))
    width = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 均值对比
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, mean_computed, width, label='实际计算值', color='steelblue')
    bars2 = ax.bar(x + width / 2, NORMALIZE_MEAN, width, label='Normalize 参数', color='coral')
    ax.set_ylabel('均值')
    ax.set_title('各通道像素均值对比')
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # 标准差对比
    ax = axes[1]
    bars1 = ax.bar(x - width / 2, std_computed, width, label='实际计算值', color='steelblue')
    bars2 = ax.bar(x + width / 2, NORMALIZE_STD, width, label='Normalize 参数', color='coral')
    ax.set_ylabel('标准差')
    ax.set_title('各通道像素标准差对比')
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pixel_statistics.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[已保存] 像素统计图: {save_path}")


def plot_augmentation_comparison(
    train_images: np.ndarray, output_dir: str, num_examples: int = 6
) -> None:
    """绘制数据增强效果对比图。

    展示原始图像与经过随机裁剪、水平翻转和归一化后的增强图像对比。

    Args:
        train_images: 训练集图像数组，形状为 (N, 3, 32, 32)。
        output_dir: 图片输出目录。
        num_examples: 展示的样本数量。
    """
    train_transform = get_transforms(train=True)

    # 反归一化用的 tensor -> PIL 转换
    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(NORMALIZE_MEAN, NORMALIZE_STD)],
        std=[1.0 / s for s in NORMALIZE_STD]
    )

    fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 2.2, 5))

    for i in range(num_examples):
        # 随机选一张训练图
        idx = np.random.randint(0, len(train_images))
        img_np = train_images[idx]  # (3, 32, 32), float32 [0,1]

        # 原始图 -> PIL
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(np.transpose(img_uint8, (1, 2, 0)))

        # 增强后的图
        img_aug = train_transform(img_pil)  # Tensor (3, 32, 32)
        img_aug_denorm = denormalize(img_aug)
        img_aug_np = img_aug_denorm.permute(1, 2, 0).numpy()
        img_aug_np = np.clip(img_aug_np, 0, 1)

        # 显示原始图
        axes[0, i].imshow(img_pil)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('原始图像', fontsize=11, loc='left')

        # 显示增强图
        axes[1, i].imshow(img_aug_np)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('增强后 (Crop+HFlip+Normalize)', fontsize=11, loc='left')

    plt.suptitle('数据增强效果对比 (反归一化显示)', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'augmentation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[已保存] 数据增强对比图: {save_path}")


def main() -> None:
    """主函数：依次执行所有数据探索可视化步骤。"""
    parser = argparse.ArgumentParser(description='CIFAR-10 数据探索可视化')
    parser.add_argument('--data_path', type=str, default='cifar-10-batches-py',
                        help='CIFAR-10 数据目录路径')
    parser.add_argument('--output_dir', type=str, default='results/data_exploration',
                        help='可视化输出目录')
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir

    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置随机种子保证可复现
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 50)
    print("CIFAR-10 数据探索可视化")
    print("=" * 50)

    # 1. 加载所有数据（用于类别分布和像素统计）
    print("\n[1/4] 加载数据...")
    train_images, train_labels = load_batches(
        data_path, [f"data_batch_{i}" for i in range(1, 6)]
    )
    test_images, test_labels = load_batches(data_path, ["test_batch"])

    # 复现验证集分层划分（与 data_loader.py 一致）
    rng = np.random.RandomState(42)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for c in range(10):
        class_indices = np.where(train_labels == c)[0]
        rng.shuffle(class_indices)
        n_val = len(class_indices) // 10
        val_indices.extend(class_indices[:n_val].tolist())
        train_indices.extend(class_indices[n_val:].tolist())
    val_labels = train_labels[val_indices]

    print(f"训练集: {len(train_indices)} 张")
    print(f"验证集: {len(val_indices)} 张")
    print(f"测试集: {len(test_labels)} 张")

    # 打印验证集各类别分布
    print("\n验证集各类别样本数:")
    for c in range(10):
        count = int(np.sum(val_labels == c))
        print(f"  类别 {c} ({CIFAR10_CLASSES[c]}): {count} 张")

    # 2. 类别分布图
    print("\n[2/4] 生成类别分布图...")
    plot_class_distribution(train_labels, val_labels, test_labels, output_dir)

    # 3. 样本图片展示
    print("\n[3/4] 生成样本图片展示...")
    plot_sample_images(train_images, train_labels, output_dir, samples_per_class=8)

    # 4. 像素统计
    print("\n[4/4] 生成像素统计图...")
    plot_pixel_statistics(train_images, output_dir)

    # 5. 数据增强对比
    print("\n[5/4] 生成数据增强对比图...")
    plot_augmentation_comparison(train_images, output_dir, num_examples=6)

    print("\n" + "=" * 50)
    print(f"所有可视化已保存到: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
