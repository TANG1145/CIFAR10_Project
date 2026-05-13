# CIFAR-10 图像分类项目

《机器学习导论》大作业 —— 题目二

## 项目结构

```
CIFAR10_Project/
├── cifar-10-batches-py/       # CIFAR-10 数据集（45K 训练 / 5K 验证 / 10K 测试）
├── checkpoints/               # 模型检查点
│   ├── mlp/
│   ├── cnn/
│   └── resnet18/
├── results/                   # 评估结果与可视化
├── src/
│   ├── data_loader.py         # 数据加载、增强、归一化
│   ├── models/
│   │   ├── mlp.py             # 多层感知机（~3.8M 参数）
│   │   ├── cnn.py             # 标准 / 改进 CNN（~3.25M / ~4.69M 参数）
│   │   └── resnet.py          # ResNet-18（~11M 参数）
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── utils.py               # 工具函数（训练循环、可视化、检查点）
├── requirements.txt           # Python 依赖
├── PROGRESS.md                # 进度总结
└── README.md                  # 本文件
```

## 环境配置

### 1. Mac 端（开发/调试）

Mac 通常没有 CUDA，用于代码编写和轻量调试：

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装CPU版PyTorch + 其他依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Win/WSL2 端（CUDA 训练）

在训练机上安装 GPU 版 PyTorch：

```bash
# 先确认 CUDA 版本
nvidia-smi

# 创建 conda 环境（推荐）
conda create -n cifar10 python=3.10 -y
conda activate cifar10

# 安装GPU版PyTorch（根据实际 CUDA 版本调整，如 cu124）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 模型与结果

| 模型 | 参数 | 测试准确率 | 说明 |
|------|------|-----------|------|
| MLP | ~3.8M | 55.57% | 3 层全连接 + BatchNorm + Dropout，基线模型 |
| CNN (Improved) | ~4.69M | 83.66% | 4 卷积块 + GAP |
| ResNet-18 | ~11M | 95.16% | 残差连接 |

## 快速开始

### 训练

```bash
# MLP
python train.py --model mlp --epochs 50 --batch_size 128 --lr 0.001 --optimizer adam --device cuda

# CNN (Improved)
python train.py --model cnn --cnn_variant improved --epochs 50 --batch_size 128 --lr 0.001 --optimizer adam --device cuda

# ResNet-18
python train.py --model resnet18 --epochs 100 --batch_size 128 --lr 0.1 --scheduler cosine --optimizer sgd --device cuda
```

### 评估

```bash
python evaluate.py --model cnn --checkpoint ../checkpoints/cnn/best_model.pth --device cuda --save_prefix cnn
```

## 参数说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|------|--------|
| `--model` | 模型类型 | `mlp` / `cnn` / `resnet18` | `cnn` |
| `--cnn_variant` | CNN 变体 | `standard` / `improved` | `improved` |
| `--epochs` | 训练轮数 | — | `200` |
| `--batch_size` | 批次大小 | — | `128` |
| `--lr` | 学习率 | — | `0.1` |
| `--weight_decay` | 权重衰减 | — | `5e-4` |
| `--optimizer` | 优化器 | `adam` / `sgd` | `sgd` |
| `--scheduler` | 学习率调度器 | `cosine` / `plateau` / `none` | `cosine` |
| `--dropout` | Dropout 率 | — | `0.3` |
| `--early_stop` | 早停耐心值 | — | `30` |
| `--device` | 计算设备 | `auto` / `cuda` / `cpu` | `auto` |
| `--resume` | 从 latest_model 恢复训练 | flag | — |
| `--seed` | 随机种子 | — | `42` |

## 输出文件

### 检查点 (`checkpoints/<model>/`)

- `best_model.pth` — 最佳验证准确率模型
- `latest_model.pth` — 最新 epoch 模型
- `config.json` — 训练配置
- `history.json` — 训练历史（loss / acc 曲线数据）

### 评估结果 (`results/`)

- `<model>_training_curves.png` — 训练/验证曲线
- `<model>_confusion_matrix.png` — 混淆矩阵
- `<model>_class_accuracy.png` — 各类别准确率
- `<model>_classification_report.txt` — 分类报告文本
- `<model>_results.json` — 结果摘要 JSON
