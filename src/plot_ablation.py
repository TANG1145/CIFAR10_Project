"""
消融实验对比图生成脚本

读取 results/ablation/ 下所有 history.json 文件，生成：
- 5 张分组对比图（各消融组内变体比较）
- 1 张总体汇总柱状图

用法： python src/plot_ablation.py
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GROUPS: Dict[str, List[Tuple[str, str, str]]] = {
    'batchnorm': [
        ('with_bn', 'With BatchNorm', 'C0'),
        ('no_bn',   'Without BatchNorm', 'C1'),
    ],
    'dropout': [
        ('dropout_0',   'Dropout = 0.0', 'C0'),
        ('dropout_0.3', 'Dropout = 0.3', 'C1'),
        ('dropout_0.5', 'Dropout = 0.5', 'C2'),
    ],
    'optimizer': [
        ('sgd',  'SGD',  'C0'),
        ('adam', 'Adam', 'C1'),
    ],
    'scheduler': [
        ('cosine',  'CosineAnnealing', 'C0'),
        ('plateau', 'ReduceLROnPlateau', 'C1'),
        ('none',    'No Scheduler', 'C2'),
    ],
    'data_aug': [
        ('no_aug',         'No Augmentation',          'C0'),
        ('normalize_only', 'Normalize Only',            'C1'),
        ('full_aug',       'Full Augmentation',         'C2'),
    ],
}

GROUP_TITLES: Dict[str, str] = {
    'batchnorm': 'Batch Normalization Ablation',
    'dropout': 'Dropout Rate Ablation',
    'optimizer': 'Optimizer Ablation',
    'scheduler': 'Learning Rate Scheduler Ablation',
    'data_aug': 'Data Augmentation Ablation',
}

RESULTS_DIR = 'results/ablation'
FINAL_DIR = 'results/final'
os.makedirs(FINAL_DIR, exist_ok=True)


def load_history(group: str, variant: str) -> dict:
    path = os.path.join(RESULTS_DIR, group, variant, 'cnn', 'history.json')
    with open(path) as f:
        return json.load(f)


def _ema_smooth(data: List[float], alpha: float = 0.3) -> List[float]:
    """指数移动平均平滑。"""
    if not data:
        return []
    smoothed = [data[0]]
    for value in data[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_group_comparison(group: str) -> Dict[str, float]:
    """
    为单个消融组绘制对比图。
    返回 {variant: best_val_acc} 字典。
    """
    variants = GROUPS[group]
    n_variants = len(variants)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    all_histories: Dict[str, dict] = {}
    best_accs: Dict[str, float] = {}
    max_epochs = 0

    for variant, label, color in variants:
        hist = load_history(group, variant)
        all_histories[variant] = hist
        best_accs[variant] = max(hist['val_acc'])
        max_epochs = max(max_epochs, len(hist['val_acc']))

    ax = axes[0]
    for variant, label, color in variants:
        hist = all_histories[variant]
        epochs = range(1, len(hist['train_acc']) + 1)
        ax.plot(epochs, hist['train_acc'], color=color, linewidth=1.2, alpha=0.35)
        ax.plot(epochs, _ema_smooth(hist['train_acc']), color=color,
                linewidth=2.5, label=label)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Accuracy (%)', fontsize=12)
    ax.set_title('Training Accuracy', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(1, max_epochs)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for variant, label, color in variants:
        hist = all_histories[variant]
        epochs = range(1, len(hist['val_acc']) + 1)
        ax.plot(epochs, hist['val_acc'], color=color, linewidth=1.2, alpha=0.35)
        smoothed = _ema_smooth(hist['val_acc'])
        line, = ax.plot(epochs, smoothed, color=color, linewidth=2.5, label=label)

        best_val = max(hist['val_acc'])
        best_ep = hist['val_acc'].index(best_val) + 1
        ax.scatter([best_ep], [best_val], color=color, s=60, zorder=5,
                   edgecolors='white', linewidths=1.2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(1, max_epochs)
    ax.grid(True, alpha=0.3)

    fig.suptitle(GROUP_TITLES[group], fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FINAL_DIR, f'ablation_{group}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

    return best_accs


def plot_overall_summary(all_best_accs: Dict[str, Dict[str, float]]) -> None:
    """
    绘制总体消融汇总柱状图。
    每个消融组下各变体的最佳验证准确率。
    """
    group_names = list(GROUPS.keys())
    n_groups = len(group_names)

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5))

    for idx, group in enumerate(group_names):
        ax = axes[idx]
        variants = GROUPS[group]
        names = [v[0] for v in variants]
        labels = [v[1] for v in variants]
        colors = [v[2] for v in variants]
        values = [all_best_accs[group][n] for n in names]

        bars = ax.bar(range(len(variants)), values, color=colors, edgecolor='black',
                      width=0.6)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
        ax.set_ylim(0, max(values) + 10)
        ax.set_ylabel('Best Validation Acc (%)', fontsize=10)
        ax.set_title(GROUP_TITLES[group].replace(' Ablation', ''), fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')



    fig.suptitle('Ablation Study Summary — Best Validation Accuracy', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(FINAL_DIR, 'ablation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def main():
    print("=" * 60)
    print("Generating ablation comparison charts...")
    print("=" * 60)

    all_best_accs: Dict[str, Dict[str, float]] = {}

    for group in GROUPS:
        print(f"\n── {group} ──")
        best_accs = plot_group_comparison(group)
        all_best_accs[group] = best_accs
        for variant, acc in best_accs.items():
            print(f"  {variant}: best_val_acc = {acc:.2f}%")

    print("\n── Overall Summary ──")
    plot_overall_summary(all_best_accs)

    print("\n" + "=" * 60)
    print("All ablation charts generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
