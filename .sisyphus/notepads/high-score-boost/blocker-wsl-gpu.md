## [2026-05-13] Blocker: WSL GPU Training Required

### Blocked Tasks
- **T10**: 运行消融实验（数据增强 + BatchNorm + Dropout + 优化器）
- **T12**: ResNet-18 增强训练（Label Smoothing + Warmup）

### Root Cause
1. **Mac 无 GPU**: 当前环境为 Mac M-series (ARM)，无 CUDA 支持
2. **PyTorch 未安装**: `python3 -m pip install torch` 因 PEP 668 被系统保护阻止
3. **WSL 远程约束**: 用户明确要求 GPU 训练在 WSL 上执行

### What Was Done
- ✅ T7: `src/run_ablation.py` 脚本已写好，支持 `--dry-run` 验证配置
- ✅ T11: `src/run_experiments.py` 脚本已写好，支持 `--dry-run` 验证配置
- ✅ T12: `src/train.py` 已添加 `--label_smoothing` 和 `--warmup_epochs` 参数

### WSL Execution Commands
```bash
# 1. 同步代码到 WSL
cd <wsl-project-dir>
git pull origin main

# 2. 安装依赖
pip install -r requirements.txt

# 3. 消融实验（预计 2-4 小时，取决于 GPU）
python src/run_ablation.py --device cuda --output_dir results/ablation

# 4. ResNet-18 增强训练（预计 2-3 小时）
python src/train.py --model resnet18 --epochs 200 --lr 0.1 \
    --label_smoothing 0.1 --warmup_epochs 5 --device cuda \
    --save_dir checkpoints/resnet18_enhanced

# 5. 超参数对比（可选，预计 4-6 小时）
python src/run_experiments.py --device cuda --output_dir results/hyperparams
```

### After WSL Training
1. 将 `results/ablation/` 和 `results/hyperparams/` 复制回 Mac
2. 运行 `python src/generate_summary.py` 更新汇总
3. 重新生成报告: `python reports/generate_notebook.py`
4. 更新计划中的 T10 和 T12 为 `[x]`

### Impact on Grading
- **代码完成情况 (25分)**: 20-23/25（缺消融实验运行结果）
- **实验效果 (25分)**: 20-23/25（缺消融数据支撑分析）
- 报告已预留消融实验章节，训练完成后填入数据即可
