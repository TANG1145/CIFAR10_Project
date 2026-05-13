"""
可视化模块

提供训练曲线、混淆矩阵、分类准确率等可视化函数，
以及结果摘要保存和分类报告打印工具。
"""

from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适合远程服务器
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """绘制训练曲线（损失和准确率）。

    Args:
        history: 训练历史字典，包含 train_loss/val_loss/train_acc/val_acc 键。
        save_path: 图片保存路径。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def _ema_smooth(data: List[float], alpha: float = 0.3) -> List[float]:
    """计算指数移动平均（EMA）平滑序列。

    Args:
        data: 原始数据列表。
        alpha: 平滑系数，越大越贴近当前值。

    Returns:
        平滑后的数据列表。
    """
    if not data:
        return []
    smoothed = [data[0]]
    for value in data[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_training_curves_improved(
    history: Dict[str, List[float]],
    save_path: str,
    smooth: bool = True,
    mark_best: bool = True,
    alpha: float = 0.3,
) -> None:
    """绘制改进版训练曲线（含 EMA 平滑、最佳 epoch 标记、train-val gap）。

    Args:
        history: 训练历史字典，包含 train_loss/val_loss/train_acc/val_acc 键。
        save_path: 图片保存路径。
        smooth: 是否绘制 EMA 平滑曲线。
        mark_best: 是否在最佳验证准确率处标记竖线。
        alpha: EMA 平滑系数。
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    ax.plot(epochs, train_loss, label='Train Loss', linewidth=1.5, alpha=0.4, color='C0')
    ax.plot(epochs, val_loss, label='Val Loss', linewidth=1.5, alpha=0.4, color='C1')
    if smooth:
        ax.plot(epochs, _ema_smooth(train_loss, alpha), label=f'Train Loss (EMA α={alpha})',
                linewidth=2.5, color='C0')
        ax.plot(epochs, _ema_smooth(val_loss, alpha), label=f'Val Loss (EMA α={alpha})',
                linewidth=2.5, color='C1')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    ax.plot(epochs, train_acc, label='Train Acc', linewidth=1.5, alpha=0.4, color='C0')
    ax.plot(epochs, val_acc, label='Val Acc', linewidth=1.5, alpha=0.4, color='C1')
    if smooth:
        ax.plot(epochs, _ema_smooth(train_acc, alpha), label=f'Train Acc (EMA α={alpha})',
                linewidth=2.5, color='C0')
        ax.plot(epochs, _ema_smooth(val_acc, alpha), label=f'Val Acc (EMA α={alpha})',
                linewidth=2.5, color='C1')

    if mark_best:
        best_epoch = int(np.argmax(val_acc)) + 1
        best_val_acc = float(val_acc[best_epoch - 1])
        ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.annotate(
            f'Best Epoch: {best_epoch}\nVal Acc: {best_val_acc:.2f}%',
            xy=(best_epoch, best_val_acc),
            xytext=(best_epoch + len(epochs) * 0.05, best_val_acc - 5),
            fontsize=10,
            color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
        )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training & Validation Accuracy', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    gap = np.array(train_acc) - np.array(val_acc)
    ax.plot(epochs, gap, linewidth=2, color='purple')
    if smooth:
        ax.plot(epochs, _ema_smooth(gap.tolist(), alpha), linewidth=2.5,
                color='darkviolet', linestyle='--', label=f'EMA α={alpha}')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.fill_between(epochs, gap, 0, alpha=0.2, color='purple')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gap (Train Acc - Val Acc) %', fontsize=12)
    ax.set_title('Train-Validation Gap', fontsize=14)
    if smooth:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved improved training curves to {save_path}")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
) -> None:
    """绘制归一化混淆矩阵。

    Args:
        y_true: 真实标签列表。
        y_pred: 预测标签列表。
        class_names: 类别名称列表。
        save_path: 图片保存路径。
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Normalized Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_class_accuracy(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
) -> None:
    """绘制各类别准确率柱状图。

    Args:
        y_true: 真实标签列表。
        y_pred: 预测标签列表。
        class_names: 类别名称列表。
        save_path: 图片保存路径。
    """
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, class_acc, color='steelblue', edgecolor='black')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy on Test Set', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    # 标注数值
    for bar, acc in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class accuracy to {save_path}")


def save_results_summary(results: Dict[str, Any], save_path: str) -> None:
    """保存评估结果摘要为 JSON 文件。

    Args:
        results: 包含评估指标的字典。
        save_path: JSON 文件保存路径。
    """
    import json
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results summary to {save_path}")


def plot_model_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str,
) -> None:
    """绘制多个模型的训练曲线对比（精度和损失）。

    Args:
        histories: 模型名称到训练历史字典的映射。
                    每个历史应包含 train_loss/val_loss/train_acc/val_acc 键。
        save_path: 图片保存路径。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Loss
    ax = axes[0]
    for i, (name, h) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        if 'val_loss' in h:
            ax.plot(h['val_loss'], label=f'{name} Val', color=c, linewidth=2)
        if 'train_loss' in h:
            ax.plot(h['train_loss'], label=f'{name} Train', color=c,
                    linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Comparison Across Models', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    for i, (name, h) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        if 'val_acc' in h:
            ax.plot(h['val_acc'], label=f'{name} Val', color=c, linewidth=2)
        if 'train_acc' in h:
            ax.plot(h['train_acc'], label=f'{name} Train', color=c,
                    linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Comparison Across Models', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison curves to {save_path}")


def plot_accuracy_bar_chart(
    accuracies: Dict[str, float],
    save_path: str,
) -> None:
    """绘制多个模型的测试准确率柱状图对比。

    Args:
        accuracies: 模型名称到测试准确率（百分比）的映射。
        save_path: 图片保存路径。
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(accuracies.keys())
    values = [accuracies[m] for m in models]

    bars = ax.bar(models, values, color=colors[:len(models)],
                  edgecolor='black', width=0.5)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Comparison Across Models', fontsize=14)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注具体数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12,
                fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy bar chart to {save_path}")


def plot_per_class_comparison(
    per_class_accs: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """绘制多个模型在各类别上的准确率分组柱状图对比。

    Args:
        per_class_accs: 模型名称到类别准确率字典的映射。
                        每个类别准确率应为 0-1 之间的比率。
        save_path: 图片保存路径。
    """
    models = list(per_class_accs.keys())
    classes = list(next(iter(per_class_accs.values())).keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    x = np.arange(len(classes))
    n_models = len(models)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        values = [per_class_accs[model][cls] * 100 for cls in classes]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model,
               color=colors[i % len(colors)], edgecolor='black')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy Comparison Across Models', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class comparison to {save_path}")


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> str:
    """打印并返回分类报告。

    Args:
        y_true: 真实标签列表。
        y_pred: 预测标签列表。
        class_names: 类别名称列表。

    Returns:
        分类报告文本字符串。
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)
    return report
