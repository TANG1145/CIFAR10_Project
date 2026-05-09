# CIFAR-10 图像分类项目

《机器学习导论》大作业 —— 题目二

## 项目结构

```
CIFAR10_Project/
├── cifar-10-batches-py/       # CIFAR-10 数据集（已下载）
├── checkpoints/               # 模型保存目录
├── results/                   # 结果图表目录
├── src/
│   ├── data_loader.py         # 数据加载与预处理
│   ├── models/
│   │   ├── mlp.py             # 多层感知机模型
│   │   └── cnn.py             # 卷积神经网络模型
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── utils.py               # 工具函数（可视化等）
├── requirements.txt           # Python依赖
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

### 2. Win 远程端（CUDA 训练）

在 Windows 训练机上安装 GPU 版 PyTorch：

```bash
# 先确认 CUDA 版本（如 CUDA 12.1）
nvidia-smi

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装GPU版PyTorch（以CUDA 12.1为例，根据实际版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **注意**：Windows 和 Mac 的 `requirements.txt` 中除 PyTorch 外其他包通用。建议在两台机器上分别安装对应版本的 torch，其余依赖一致。

## 快速开始

### 在 Mac 上轻量测试（CPU，少量 epoch）

```bash
source venv/bin/activate
cd src
python train.py --model cnn --epochs 2 --batch_size 64 --device cpu
```

### 在 Win 远程机上正式训练（CUDA）

```bash
venv\Scripts\activate
cd src
python train.py --model cnn --epochs 50 --batch_size 128 --device cuda
python train.py --model mlp --epochs 50 --batch_size 128 --device cuda
```

### 评估模型

```bash
python evaluate.py --model cnn --checkpoint ../checkpoints/cnn_best.pth --device cuda
```

## 跨平台/远程开发建议

1. **代码同步**：使用 Git 仓库或 VSCode SFTP 插件同步 `src/` 代码
2. **数据集**：Win 端也下载一份到相同相对路径，或通过网络共享
3. **路径兼容**：代码中使用 `os.path.join`/`pathlib.Path`，已保证跨平台
4. **设备自动回退**：脚本默认自动检测 CUDA，也可用 `--device` 指定

## 关键参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型类型：`mlp` 或 `cnn` | `--model cnn` |
| `--epochs` | 训练轮数 | `--epochs 50` |
| `--batch_size` | 批次大小 | `--batch_size 128` |
| `--lr` | 学习率 | `--lr 0.001` |
| `--device` | 计算设备：`cuda`/`cpu`/`auto` | `--device cuda` |
| `--data_path` | 数据集路径 | `--data_path ../cifar-10-batches-py` |
