"""
模型评估脚本
加载训练好的模型，在测试集上评估并生成可视化结果
"""

import os
import sys
import argparse
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_loader import get_data_loaders, CIFAR10_CLASSES
from models import create_model
from checkpoint import load_checkpoint
from visualization import (plot_confusion_matrix, plot_class_accuracy,
                           print_classification_report, save_results_summary)
from device import get_device


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        包含所有评估配置参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description='CIFAR-10 Evaluation')

    default_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cifar-10-batches-py")
    parser.add_argument('--data_path', type=str, default=default_data_path)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn', 'resnet18'])
    parser.add_argument('--cnn_variant', type=str, default='improved',
                        choices=['standard', 'improved'])
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_prefix', type=str, default='model',
                        help='结果文件前缀')

    return parser.parse_args()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_prefix: str = 'model',
) -> float:
    """评估模型并在测试集上收集预测结果，生成可视化报告。

    Args:
        model: 要评估的模型。
        dataloader: 测试数据加载器。
        device: 计算设备。
        save_prefix: 结果文件前缀。

    Returns:
        测试集准确率（百分比）。
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 结果目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 混淆矩阵
    cm_path = os.path.join(results_dir, f'{save_prefix}_confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, CIFAR10_CLASSES, cm_path)

    # 各类别准确率
    acc_path = os.path.join(results_dir, f'{save_prefix}_class_accuracy.png')
    plot_class_accuracy(all_labels, all_preds, CIFAR10_CLASSES, acc_path)

    # 分类报告
    report = print_classification_report(all_labels, all_preds, CIFAR10_CLASSES)
    report_path = os.path.join(results_dir, f'{save_prefix}_classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

    # 保存结果摘要
    results = {
        'model': save_prefix,
        'test_accuracy': accuracy,
        'test_precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'test_recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'test_f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'per_class_accuracy': {
            cls: float(acc) for cls, acc in zip(CIFAR10_CLASSES,
            np.diag(confusion_matrix(all_labels, all_preds)) /
            np.array(confusion_matrix(all_labels, all_preds)).sum(axis=1))
        }
    }
    summary_path = os.path.join(results_dir, f'{save_prefix}_results.json')
    save_results_summary(results, summary_path)

    return accuracy


def main() -> None:
    """主评估函数。"""
    args = parse_args()
    device = get_device(args.device)

    # 加载数据
    print("Loading data...")
    _, _, test_loader = get_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 构建模型
    print(f"Building {args.model.upper()} model...")
    model = create_model(args.model, variant=args.cnn_variant, dropout_rate=args.dropout)
    model = model.to(device)

    # 加载权重
    print(f"Loading checkpoint from {args.checkpoint}")
    epoch, best_acc = load_checkpoint(model, None, args.checkpoint, device)
    print(f"Checkpoint: Epoch {epoch}, Best Val Acc: {best_acc:.2f}%")

    # 评估
    print("\n" + "="*60)
    print("Evaluation Start")
    print("="*60)
    test_acc = evaluate_model(model, test_loader, device, save_prefix=args.save_prefix)
    print("="*60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
