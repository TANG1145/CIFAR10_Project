# CIFAR-10 图像分类项目

《机器学习导论》大作业 —— 题目二

## 项目简介

在 CIFAR-10 数据集上实现并对比三种模型架构（MLP、CNN、ResNet-18），通过消融实验分析 Batch Normalization、Dropout、优化器选择、学习率调度和数据增强对模型性能的影响。

**最终成绩：ResNet-18 测试准确率 95.16%**（增强训练，100 轮）

## 项目结构

```
CIFAR10_Project/
├── cifar-10-batches-py/       # CIFAR-10 数据集
├── checkpoints/               # 模型检查点（.gitignore 排除）
├── results/                   # 实验结果与可视化
│   ├── mlp_retrain/           # MLP 重训数据（Adam, lr=0.001, 200ep, 60.51%）
│   ├── cnn_retrain/           # CNN 重训数据（SGD, lr=0.1, 200ep, 91.11%）
│   ├── resnet18_baseline/     # ResNet-18 基线（SGD+Cosine, 200ep, 95.05%）
│   ├── resnet18_enhanced/     # ResNet-18 增强（Label Smoothing+Warmup, 100ep, 95.16%）
│   ├── ablation/              # 5 组消融实验数据
│   └── final/                 # 最终汇总图片
├── src/
│   ├── models/
│   │   ├── mlp.py             # 多层感知机（~3.8M 参数）
│   │   ├── cnn.py             # 标准 / 改进 CNN（~3.25M / ~4.69M 参数）
│   │   └── resnet.py          # ResNet-18（~11M 参数）
│   ├── train.py               # 统一训练脚本
│   ├── evaluate.py            # 评估脚本
│   ├── data_loader.py         # 数据加载与增强
│   ├── visualization.py       # 可视化工具
│   ├── generate_summary.py    # 汇总所有图表
│   ├── run_ablation.py        # 消融实验框架
│   ├── plot_ablation.py       # 消融实验绘图
│   └── utils.py               # 工具函数
├── tests/                     # 单元测试
├── reports/
│   ├── report.tex  # LaTeX 报告源文件（25 页）
│   ├── report.pdf  # 报告 pdf
│   ├── references.bib         # 参考文献
│   └── images/                # 报告用图
├── .vscode/settings.json      # VS Code 工作区配置
├── requirements.txt           # Python 依赖
└── README.md                  # 本文件
```

## 模型与结果

| 模型              | 参数   | 测试准确率 | 训练配置                                 |
| ----------------- | ------ | ---------- | ---------------------------------------- |
| MLP               | ~3.8M  | 60.51%     | Adam, lr=0.001, 200ep                    |
| CNN (Improved)    | ~4.69M | 91.11%     | SGD, lr=0.1, 200ep                       |
| ResNet-18（基线） | ~11M   | 95.05%     | SGD+Cosine, 200ep                        |
| ResNet-18（增强） | ~11M   | **95.16%** | SGD+Cosine+Label Smoothing+Warmup, 100ep |

### 消融实验（50 轮，CNN）

| 实验组    | 最佳配置 | 准确率 | 对比基线 |
| --------- | -------- | ------ | -------- |
| BatchNorm | with_bn  | 89.88% | +79.88%  |
| Scheduler | cosine   | 89.80% | +9.26%   |
| Data Aug  | full_aug | 89.58% | +7.46%   |
| Dropout   | 0.3      | 90.62% | +2.54%   |
| Optimizer | adam     | 89.58% | +0.06%   |

## 环境配置

### Mac 端（开发/调试）

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### WSL2 / Linux 端（CUDA 训练）

```bash
conda create -n cifar10 python=3.10 -y
conda activate cifar10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 快速开始

### 训练

```bash
cd src

# MLP
python train.py --model mlp --epochs 200 --batch_size 128 --lr 0.001 --optimizer adam

# CNN (Improved)
python train.py --model cnn --cnn_variant improved --epochs 200 --batch_size 128 --lr 0.1 --optimizer sgd

# ResNet-18 基线
python train.py --model resnet18 --epochs 200 --batch_size 128 --lr 0.1 --optimizer sgd --scheduler cosine

# ResNet-18 增强训练（Label Smoothing + Warmup）
python train.py --model resnet18 --epochs 100 --batch_size 128 --lr 0.1 --optimizer sgd --scheduler cosine --label_smoothing 0.1 --warmup_epochs 5
```

### 消融实验

```bash
python run_ablation.py --experiment batchnorm
python run_ablation.py --experiment dropout
python run_ablation.py --experiment optimizer
python run_ablation.py --experiment scheduler
python run_ablation.py --experiment data_aug
```

### 生成所有图表

```bash
python generate_summary.py
```

### 评估

```bash
python evaluate.py --model cnn --checkpoint ../checkpoints/cnn/best_model.pth --device cuda
```

## 参数说明

| 参数                | 说明           | 可选值                        | 默认值     |
| ------------------- | -------------- | ----------------------------- | ---------- |
| `--model`           | 模型类型       | `mlp` / `cnn` / `resnet18`    | `cnn`      |
| `--cnn_variant`     | CNN 变体       | `standard` / `improved`       | `improved` |
| `--epochs`          | 训练轮数       | —                             | `200`      |
| `--batch_size`      | 批次大小       | —                             | `128`      |
| `--lr`              | 学习率         | —                             | `0.1`      |
| `--weight_decay`    | 权重衰减       | —                             | `5e-4`     |
| `--optimizer`       | 优化器         | `adam` / `sgd`                | `sgd`      |
| `--scheduler`       | 学习率调度器   | `cosine` / `plateau` / `none` | `none`     |
| `--label_smoothing` | 标签平滑系数   | 0.0~1.0                       | `0.0`      |
| `--warmup_epochs`   | Warmup 轮数    | —                             | `0`        |
| `--dropout`         | Dropout 比率   | 0.0~0.5                       | `0.3`      |
| `--use_bn`          | 使用 BatchNorm | —                             | `True`     |
| `--device`          | 设备           | `cpu` / `cuda`                | `cuda`     |
| `--seed`            | 随机种子       | —                             | `42`       |

## 报告

LaTeX 报告位于 `reports/report_overleaf_images.tex`，共 25 页，包含：
- 封面与独创性声明
- 5 个章节：引言、数据探索、模型架构、实验结果、结论
- 28 张可视化图表
- 5 篇参考文献

## 测试

```bash
pytest tests/ -v
```
