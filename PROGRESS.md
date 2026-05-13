# CIFAR-10 图像分类项目 —— 进度总结

> 更新时间：2026/05/12
> 课程：《机器学习导论》大作业 —— 题目二

---

## 一、项目概况

在 CIFAR-10 数据集（32x32 RGB，10 个类别）上构建并对比 MLP、CNN 和 ResNet-18 三种模型，完成图像分类任务，并撰写实验报告。

**开发模式**：Mac（代码编写 + 轻量调试）+ Windows/WSL2（CUDA 远程训练）

---

## 二、环境配置状态

### Mac 端（本地开发）
- Python 虚拟环境已创建（`venv`）
- CPU 版 PyTorch 已安装
- 所有依赖已满足

### Win/WSL2 端（CUDA 训练）
- Miniconda 已安装
- Conda 环境 `cifar10` 已创建
- PyTorch 2.11 + CUDA 12.4 已通过 pip 安装
- GPU：NVIDIA RTX 4050，CUDA 版本 13.2（兼容 cu124）
- **环境配置已完成，模型训练可用**

### 已解决的环境问题
| 问题 | 解决方式 |
|------|----------|
| PyTorch pip 下载超时 | 换用 conda 安装 |
| `pytorch-cuda=12.6` 在 conda 中不存在 | 降级到 `pytorch-cuda=12.4` |
| C 盘空间不足 | 清理磁盘空间后重试 |
| conda 下载冻结 | `wsl --shutdown` 重启后恢复 |
| `iJIT_NotifyEvent` 未定义符号 | 卸载 conda PyTorch，改用 pip 安装 |
| `verbose` 参数在 `ReduceLROnPlateau` 中不存在 | 移除 `verbose=True` 参数 |

---

## 三、项目文件结构

```
CIFAR10_Project/
├── cifar-10-batches-py/       # CIFAR-10 数据集（45K 训练 / 5K 验证 / 10K 测试）
├── checkpoints/               # 模型检查点
│   ├── mlp/
│   │   ├── best_model.pth
│   │   ├── latest_model.pth
│   │   ├── config.json
│   │   └── history.json
│   └── cnn/
│       ├── best_model.pth
│       ├── latest_model.pth
│       ├── config.json
│       └── history.json
├── results/                   # 评估结果与可视化
│   ├── mlp_*.png / mlp_*.txt / mlp_results.json
│   └── cnn_*.png / cnn_*.txt / cnn_results.json
├── src/
│   ├── data_loader.py         # 数据加载、增强、归一化
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py             # 3 层 MLP（~3.8M 参数）
│   │   ├── cnn.py             # Standard / Improved CNN（~3.25M / ~4.69M）
│   │   └── resnet.py          # ResNet-18（~11M 参数）
│   ├── train.py               # 训练脚本（支持 CLI 参数）
│   ├── evaluate.py            # 评估脚本（生成混淆矩阵、分类报告等）
│   └── utils.py               # 工具函数（训练循环、可视化、检查点）
├── .vscode/settings.json       # VSCode 配置
├── requirements.txt           # Python 依赖
├── README.md                  # 项目说明
└── PROGRESS.md                # 本文件
```

---

## 四、已完成的工作

### 1. 代码开发
- [x] 数据加载模块（`data_loader.py`）
  - 本地 pickle 格式 CIFAR-10 加载
  - 训练集划分为 45,000 训练 / 5,000 验证
  - 数据增强：RandomCrop(32, padding=4) + RandomHorizontalFlip
  - 归一化：mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
- [x] MLP 模型（`mlp.py`）
  - 3 层全连接网络，含 BatchNorm1d + ReLU + Dropout
- [x] CNN 模型（`cnn.py`）
  - Standard CNN：3 个卷积块，~3.25M 参数
  - Improved CNN：4 个卷积块 + GAP，~4.69M 参数
- [x] ResNet-18 模型（`resnet.py`）
  - 基于 pytorch-cifar 参考实现
  - BasicBlock + 残差连接，~11M 参数
- [x] 训练脚本（`train.py`）
  - 支持 MLP / CNN / ResNet18
  - 支持 Adam / SGD 优化器
  - 支持 CosineAnnealingLR / ReduceLROnPlateau / 无调度器
  - 早停机制、检查点保存、训练曲线绘制
- [x] 评估脚本（`evaluate.py`）
  - 测试集准确率、精确率、召回率、F1
  - 混淆矩阵、各类别准确率柱状图
  - 分类报告文本、JSON 结果摘要

### 2. 模型训练与评估

| 模型 | 变体 | 测试准确率 | 关键特点 |
|------|------|-----------|---------|
| MLP | - | **56.51%** | 3 层全连接，基线模型 |
| CNN | improved | **80.17%** | 4 卷积块 + GAP，最佳实用模型 |
| ResNet-18 | - | 待训练 | 残差连接，预期 90%+ |

### 3. 可视化结果（已生成）
- `mlp_training_curves.png` —— MLP 训练/验证损失与准确率曲线
- `mlp_confusion_matrix.png` —— MLP 混淆矩阵
- `mlp_class_accuracy.png` —— MLP 各类别准确率
- `cnn_training_curves.png` —— CNN 训练/验证损失与准确率曲线
- `cnn_confusion_matrix.png` —— CNN 混淆矩阵
- `cnn_class_accuracy.png` —— CNN 各类别准确率

### 4. Git 同步
- [x] GitHub 仓库已创建
- [x] 代码已推送到远程（含 ResNet-18 和 `verbose` 修复）
- [ ] **WSL2 端 git pull 因代理问题尚未成功**

---

## 五、待办事项

### 高优先级（核心作业要求）
- [ ] **训练 ResNet-18**：在 WSL2 上运行训练，预期达到 90%+ 测试准确率
- [ ] **评估 ResNet-18**：生成对应的混淆矩阵、分类报告、训练曲线
- [ ] **撰写实验报告**：包含以下 5 个部分
  1. 数据预处理与增强方法说明
  2. 模型结构描述（MLP / CNN / ResNet）
  3. 训练过程与超参数设置
  4. 实验结果对比与分析（准确率、混淆矩阵、各类别表现）
  5. 结论与展望

### 中优先级（完善工作）
- [ ] 修复 WSL2 Git 代理配置（Clash 7897 端口连接问题）
- [ ] 从 WSL2 下载训练结果（检查点、可视化图）到 Mac 端
- [ ] 整理所有实验结果，为报告撰写准备素材

### 低优先级（可选优化）
- [ ] 尝试数据增强策略对比（如 Cutout、AutoAugment）
- [ ] 尝试不同的学习率调度策略
- [ ] 模型集成（Ensemble）实验

---

## 六、已知问题与状态

| 问题 | 状态 | 说明 |
|------|------|------|
| WSL2 Git 无法连接代理 | **进行中** | Clash "Allow LAN" 已开启，但 WSL2 连接 `7897` 端口失败。尝试过的 IP：10.21.199.254、10.21.199.212、100.123.67.85、192.168.142.1 均失败 |
| 临时同步方案 | **可用** | Mac 端代码已推送到 GitHub；WSL2 可手动复制代码文件，或使用 `rsync` 同步 |

### WSL2 代理调试记录
- Windows `ipconfig` 显示的 IP：100.123.67.85（Tailscale）、192.168.142.1、192.168.64.1、10.21.199.212
- 错误类型：`gnutls_handshake() failed`、`Failed to connect to <IP> port 7897`
- 可能原因：Clash 未监听在 0.0.0.0、WSL2 网络隔离、防火墙拦截

---

## 七、下一步建议

由于 WSL2 Git 代理问题短期内难以解决，建议采用以下并行策略：

1. **立即开始 ResNet-18 训练**
   - 在 WSL2 上直接创建/编辑 `src/models/resnet.py`（文件内容简单，可手动复制）
   - 运行训练命令：
     ```bash
     cd src
     python train.py --model resnet18 --epochs 100 --batch_size 128 --lr 0.001 --scheduler cosine --optimizer adam --device cuda
     ```

2. **同步方面**
   - Mac 端继续作为代码主仓库，正常 push 到 GitHub
   - WSL2 端暂用手动方式更新代码（关键文件只有 `resnet.py` 和 `train.py` 的少量改动）

3. **报告撰写**
   - Mac 端可以直接使用已有的 `results/` 中的 MLP 和 CNN 结果
   - ResNet-18 结果待训练完成后从 WSL2 复制到 Mac

---

## 八、关键命令速查

### 训练
```bash
# MLP
python train.py --model mlp --epochs 50 --batch_size 128 --lr 0.001 --device cuda

# CNN (improved)
python train.py --model cnn --cnn_variant improved --epochs 50 --batch_size 128 --lr 0.001 --device cuda

# ResNet-18
python train.py --model resnet18 --epochs 100 --batch_size 128 --lr 0.001 --scheduler cosine --device cuda
```

### 评估
```bash
python evaluate.py --model cnn --checkpoint ../checkpoints/cnn/best_model.pth --device cuda --save_prefix cnn
```

### Git（Mac 端正常）
```bash
git add .
git commit -m "..."
git push origin main
```
