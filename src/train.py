"""
模型训练脚本
支持 MLP 和 CNN，支持 Mac CPU / Win CUDA 跨平台训练
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from data_loader import get_data_loaders, CIFAR10_CLASSES
from models.mlp import create_mlp
from models.cnn import create_cnn
from utils import train_epoch, validate_epoch, save_checkpoint, plot_training_curves


def parse_args():
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
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='模型类型: mlp 或 cnn')
    parser.add_argument('--cnn_variant', type=str, default='improved',
                        choices=['standard', 'improved'],
                        help='CNN 变体 (仅 CNN 有效)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 率')

    # 训练
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减 (L2正则化)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['plateau', 'cosine', 'none'],
                        help='学习率调度器')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='早停耐心值 (0 表示禁用)')

    # 系统
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备: auto/cuda/cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 输出
    parser.add_argument('--save_dir', type=str, default=None,
                        help='模型保存目录 (默认 ../checkpoints/模型名/)')

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子保证可复现"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg):
    """获取计算设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    return device


def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备
    device = get_device(args.device)

    # 保存路径
    if args.save_dir is None:
        args.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", args.model
        )
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {args.save_dir}")

    # 保存配置
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    # 数据
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 模型
    print(f"\nBuilding {args.model.upper()} model...")
    if args.model == 'mlp':
        model = create_mlp(dropout_rate=args.dropout)
    else:
        model = create_cnn(variant=args.cnn_variant, dropout_rate=args.dropout)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay, nesterov=True)

    # 学习率调度
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None

    # 训练循环
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 学习率调整
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        if args.scheduler == 'plateau':
            scheduler.step(val_acc)
        elif args.scheduler == 'cosine':
            scheduler.step()

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_acc, best_path)
            print(f"Best model saved! (Val Acc: {best_acc:.2f}%)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 保存最新模型
        latest_path = os.path.join(args.save_dir, 'latest_model.pth')
        save_checkpoint(model, optimizer, epoch, best_acc, latest_path)

        # 早停
        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # 绘制训练曲线
    results_dir = os.path.join(os.path.dirname(args.save_dir), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f'{args.model}_training_curves.png')
    plot_training_curves(history, plot_path)

    print("\n" + "="*60)
    print("Training Complete")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.save_dir}")
    print("="*60)

    # 测试集最终评估
    print("\nEvaluating on test set...")
    from evaluate import evaluate_model
    test_acc = evaluate_model(model, test_loader, device, save_prefix=args.model)
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
