"""
模型训练脚本
支持 MLP 和 CNN，支持 Mac CPU / Win CUDA 跨平台训练
"""

import os
import sys
import argparse
import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from data_loader import get_data_loaders, CIFAR10_CLASSES
from torchvision import transforms
from models import create_model
from training import train_epoch, validate_epoch
from checkpoint import save_checkpoint, load_checkpoint
from visualization import plot_training_curves
from device import get_device
from evaluate import evaluate_model


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        包含所有训练配置参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')

    # 数据
    default_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cifar-10-batches-py")
    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help='CIFAR-10 数据集路径')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 模型
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['mlp', 'cnn', 'resnet18'],
                        help='模型类型: mlp, cnn 或 resnet18')
    parser.add_argument('--cnn_variant', type=str, default='improved',
                        choices=['standard', 'improved'],
                        help='CNN 变体 (仅 CNN 有效)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 率')
    parser.add_argument('--use_bn', type=bool, default=True, help='Use BatchNorm in CNN')

    # 训练
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减 (L2正则化)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD 动量 (仅 SGD 有效)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['plateau', 'cosine', 'none'],
                        help='学习率调度器')
    parser.add_argument('--early_stop', type=int, default=30,
                        help='早停耐心值 (0 表示禁用)')

    # 系统
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备: auto/cuda/cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='从 latest_model.pth 恢复训练')

    # 输出
    parser.add_argument('--save_dir', type=str, default=None,
                        help='模型保存目录 (默认 ../checkpoints/模型名/)')

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子以保证结果可复现。

    Args:
        seed: 随机种子值。
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def run_training(
    args: argparse.Namespace,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
) -> Dict[str, Any]:
    set_seed(args.seed)
    device = get_device(args.device)

    if args.save_dir is None:
        args.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", args.model
        )
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {args.save_dir}")

    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Config saved to {config_path}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    print(f"\nBuilding {args.model.upper()} model...")
    model = create_model(args.model, variant=args.cnn_variant, dropout_rate=args.dropout, use_bn=getattr(args, 'use_bn', True))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        momentum = getattr(args, 'momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum,
                              weight_decay=args.weight_decay, nesterov=(momentum > 0))

    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None

    start_epoch = 1
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    epochs_no_improve = 0

    if args.resume:
        resume_path = os.path.join(args.save_dir, 'latest_model.pth')
        history_path = os.path.join(args.save_dir, 'history.json')
        if os.path.exists(resume_path):
            print(f"\nResuming from {resume_path}")
            start_epoch, best_acc = load_checkpoint(model, optimizer, resume_path, device)
            start_epoch += 1
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
            print(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")
        else:
            print(f"Checkpoint not found: {resume_path}, starting from scratch")

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        if args.scheduler == 'plateau':
            scheduler.step(val_acc)
        elif args.scheduler == 'cosine':
            scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_acc, best_path)
            print(f"Best model saved! (Val Acc: {best_acc:.2f}%)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        latest_path = os.path.join(args.save_dir, 'latest_model.pth')
        save_checkpoint(model, optimizer, epoch, best_acc, latest_path)

        history_path = os.path.join(args.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    history_path = os.path.join(args.save_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    results_dir = os.path.join(os.path.dirname(args.save_dir), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f'{args.model}_training_curves.png')
    plot_training_curves(history, plot_path)

    print("\n" + "="*60)
    print("Training Complete")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.save_dir}")
    print("="*60)

    print("\nEvaluating on test set...")
    test_acc = evaluate_model(model, test_loader, device, save_prefix=args.model)
    print(f"Test Accuracy: {test_acc:.2f}%")

    return history


def main() -> None:
    """主训练函数。"""
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
