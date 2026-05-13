

## 2026-05-13 data_exploration.py 实现

### 成功要点
- 复用了 `data_loader.py` 的 `CIFAR10_CLASSES` 和 `get_transforms()`
- 使用 `load_batches` 直接读取 pickle 文件，避免 transform 干扰统计
- 验证集划分完全复现 `data_loader.py` 的 `torch.randperm` + seed=42 逻辑
- 中文字体自动探测：优先尝试 Arial Unicode MS / Hiragino Sans GB / PingFang HK / Hei

### 遇到的问题
1. **matplotlib 中文字体缺失** → 通过 `matplotlib.rcParams['font.family']` 动态设置系统可用字体
2. **IndexError in sample_images** → 最初把 `samples_per_class // 2` 误当列数，应为 `samples_per_class` 列（每类一行，每行8张图）
3. **NumPy 2.4 pickle 警告** → `pickle.load` 产生 `VisibleDeprecationWarning`，与 `data_loader.py` 现有代码一致，未修改

### 生成的可视化
- `class_distribution.png` (58K) — 训练/验证/测试集类别分布
- `sample_images.png` (352K) — 每类8张随机样本
- `pixel_statistics.png` (59K) — 各通道均值/标准差与 Normalize 参数对比
- `augmentation_comparison.png` (76K) — 原始图 vs RandomCrop+HFlip+Normalize 增强后

### 依赖
在项目根目录创建了 `venv/` 虚拟环境，安装了 `matplotlib torch torchvision numpy`。
