# CIFAR-10 高分提升计划

## TL;DR

> **Quick Summary**: 弥补项目与高分要求的4维度差距（代码完成/代码规范/实验效果/报告规范），通过重构代码、补充消融实验和超参数对比、改进可视化、最终撰写报告，冲刺92+分。
> 
> **Deliverables**:
> - 重构后的代码库（类型注解、消除重复、模块化）
> - 数据探索可视化（类别分布、样本展示、像素统计）
> - 消融实验代码 + 结果（数据增强、BatchNorm、Dropout、优化器）
> - 超参数对比实验代码 + 结果（学习率、batch size、调度器）
> - 改进的可视化（平滑曲线、best epoch 标注、多模型对比图）
> - 实验脚本（一键批量跑实验）
> - 单元测试
> - 完整 PDF 报告
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves + final report
> **Critical Path**: Task 1→2→5→7→10→13→14→15→F1-F4
> 
> **WSL Constraint**: Wave 1 全部在 Mac 本地执行（纯代码重构，无需 GPU）。Wave 2+ 的实验脚本在 Mac 本地编写，训练运行需要同步代码到 WSL 远程执行（GPU 训练）。

---

## Context

### Original Request
用户希望根据作业评分标准弥补CIFAR-10项目与高分要求的差距，冲击92-100分。报告最后写。

### Interview Summary
**Key Discussions**:
- 评分4维度各25分：代码完成情况、代码规范性、实验效果、报告规范性
- 当前预估：代码完成20-22分、代码规范18-22分、实验效果18-22分、报告0分
- 用户明确要求报告最后写，先做代码和实验改进

**Research Findings**:
- ResNet-18 at 95.16% 处于标准范围(93-95%)，已达标
- 最有效消融：数据增强(+3-5%)、BatchNorm(+7%)、优化器对比
- 训练曲线存在严重问题（锯齿、无平滑、无best epoch标注）
- 验证集使用随机划分而非分层抽样
- `get_device()` 在 train.py 和 evaluate.py 中重复定义
- `utils.py` 混合3种职责
- 无类型注解、无单元测试、`__init__.py` 空文件

### Metis Review
**Identified Gaps** (addressed):
- 时间预算和GPU可用性不明 → 假设有足够时间，实验设计考虑GPU时间
- 报告格式要求不明 → 按作业文档的5部分结构撰写
- 是否需要重训已有模型 → 不重训ResNet-18(95.16%已好)，仅跑新消融实验
- 会否过度工程化 → 设置guardrails防止scope creep

---

## Work Objectives

### Core Objective
将项目从当前约56-81分提升至92+分，覆盖评分4维度的所有高分要求。

### Concrete Deliverables
- `src/utils.py` → 拆分为 `training.py` + `visualization.py` + `checkpoint.py`
- `src/device.py` → 统一设备管理，消除重复
- `src/models/__init__.py` → 导出3个模型工厂
- 所有函数添加类型注解
- `src/data_exploration.py` → 数据探索可视化脚本
- `src/run_ablation.py` → 消融实验批量运行脚本
- `src/run_hyperparam_search.py` → 超参数对比实验脚本
- `results/` 下新增所有实验结果文件
- 改进的可视化（平滑曲线、best epoch、对比图）
- `tests/` 单元测试目录
- 完整 PDF 报告

### Definition of Done
- [ ] 所有函数有类型注解和docstring
- [ ] 无重复代码（DRY）
- [ ] 数据探索图（类别分布、样本展示）存在于 `results/`
- [ ] 消融实验结果存在于 `results/ablation/`
- [ ] 超参数对比结果存在于 `results/hyperparam/`
- [ ] 改进后的训练曲线图（平滑+best epoch）存在于 `results/`
- [ ] 多模型对比图存在于 `results/`
- [ ] `pytest tests/` 全部通过
- [ ] PDF 报告包含5个必要部分

### Must Have
- 代码重构完成且现有功能不受影响
- 至少4组消融实验结果（数据增强、BatchNorm、Dropout、优化器）
- 至少3组超参数对比结果
- 数据探索可视化脚本和生成图
- 改进的训练曲线可视化
- 完整PDF报告（5部分）

### Must NOT Have (Guardrails)
- ❌ 不重新训练已有最佳模型（ResNet-18 95.16%保持不变）
- ❌ 不添加超出作业范围的功能（如TensorBoard、WandB集成）
- ❌ 不引入不必要的抽象层或过度工程化
- ❌ 不修改已有模型的架构定义（mlp.py/cnn.py/resnet.py内的网络结构）
- ❌ 不破坏Mac/Win/WSL跨平台兼容性
- ❌ 不写超过5页的报告（精炼为主）
- ❌ 不添加CNN standard变体的训练（作业只要求MLP和CNN对比）
- ❌ Wave 2+ 任务不得在 Mac 上运行 GPU 训练（所有训练必须在 WSL 上执行）

### Execution Environment
- **Wave 1 (Task 1-6)**: Mac 本地执行（纯代码重构+脚本编写，无需 GPU）
- **Wave 2+ 实验脚本 (Task 7, 8, 9, 11)**: Mac 本地编写脚本
- **Wave 2+ 实验运行 (Task 10, 12)**: 代码同步到 WSL 后在 GPU 上执行
- **Wave 3-5**: Mac 编写 + WSL 训练混合
- **代码同步方式**: git push/pull 或手动文件复制

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO（当前无测试框架）
- **Automated tests**: YES (Tests-after) — 先重构代码，再为关键函数补测试
- **Framework**: pytest
- **If TDD**: N/A (tests-after)

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python scripts**: Use Bash — `python script.py` and check exit code + output files
- **Tests**: Use Bash — `pytest tests/ -v` and verify pass
- **Visualizations**: Use Bash — run script, check output PNG/JSON exists in results/

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation + independent code improvements) [MAC LOCAL]:
├── Task 1: 拆分utils.py为模块化结构 [quick]
├── Task 2: 统一设备管理+消除重复代码 [quick]
├── Task 3: 补充类型注解和docstring [quick]
├── Task 4: 完善models/__init__.py导出 [quick]
├── Task 5: 数据探索可视化脚本 [unspecified-high]
└── Task 6: 修复验证集为分层抽样 [quick]

Wave 2 (After Wave 1 - experiments, depends on refactored code) [MAC script + WSL training]:
├── Task 7: 消融实验框架脚本 [deep] — Mac本地编写
├── Task 8: 改进训练曲线可视化 [unspecified-high] — Mac本地
├── Task 9: 多模型对比可视化脚本 [quick] — Mac本地
└── Task 10: 运行消融实验(数据增强+BatchNorm+Dropout) [deep] — WSL GPU执行

Wave 3 (After Wave 2 - hyperparameter + remaining experiments) [MAC script + WSL training]:
├── Task 11: 超参数对比实验脚本 [unspecified-high] — Mac本地编写
├── Task 12: 运行超参数对比实验 [deep] — WSL GPU执行
└── Task 13: 单元测试 [unspecified-high] — Mac本地

Wave 4 (After experiments done - report):
└── Task 14: 完整PDF报告 [writing]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1→8→10→12→14→F1-F4
Max Concurrent: 6 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | - | 7, 8, 9 |
| 2 | 1 | 7, 10 |
| 3 | 1 | 13 |
| 4 | - | 7 |
| 5 | - | 14 |
| 6 | 1 | 10 |
| 7 | 1, 2, 4 | 10 |
| 8 | 1 | 14 |
| 9 | 1 | 14 |
| 10 | 2, 6, 7 | 14 |
| 11 | 7 | 12 |
| 12 | 11 | 14 |
| 13 | 3 | F2 |
| 14 | 5, 8, 9, 10, 12 | F1, F4 |

### Agent Dispatch Summary

- **Wave 1**: **6** — T1-T4 → `quick`, T5 → `unspecified-high`, T6 → `quick`
- **Wave 2**: **4** — T7 → `deep`, T8 → `unspecified-high`, T9 → `quick`, T10 → `deep`
- **Wave 3**: **3** — T11 → `unspecified-high`, T12 → `deep`, T13 → `unspecified-high`
- **Wave 4**: **1** — T14 → `writing`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. 拆分 utils.py 为模块化结构

  **What to do**:
  - 将 `src/utils.py` 拆分为3个独立模块：
    - `src/training.py` — `train_epoch()`, `validate_epoch()` 函数（训练/验证循环）
    - `src/visualization.py` — `plot_training_curves()`, `plot_confusion_matrix()`, `plot_class_accuracy()`, `save_results_summary()`, `print_classification_report()` 函数（可视化）
    - `src/checkpoint.py` — `save_checkpoint()`, `load_checkpoint()` 函数（检查点管理）
  - 每个新模块添加类型注解和docstring
  - 更新 `src/train.py` 和 `src/evaluate.py` 中的 import 语句（从 `from utils import ...` 改为对应模块的 import）
  - 保留 `src/utils.py` 作为兼容性import桥（import from training, visualization, checkpoint），向后兼容
  - 确保拆分后所有现有功能正常工作

  **Must NOT do**:
  - 不修改训练循环的算法逻辑
  - 不改变任何函数的参数签名
  - 不删除原始 utils.py（保留兼容桥）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 拆分文件是结构清晰的操作，逻辑简单
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `refactor`: 重构不是核心技能

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5, 6)
  - **Blocks**: Tasks 7, 8, 9
  - **Blocked By**: None

  **References**:
  **Pattern References**:
  - `src/utils.py:1-175` — 当前所有函数，需要分类到3个新模块
  - `src/train.py:19` — `from utils import train_epoch, validate_epoch, save_checkpoint, load_checkpoint, plot_training_curves`
  - `src/evaluate.py:18-19` — `from utils import load_checkpoint, plot_confusion_matrix, plot_class_accuracy, print_classification_report, save_results_summary`

  **Why Each Reference Matters**:
  - `src/utils.py` 完整内容决定了哪些函数归哪个模块
  - `train.py` 和 `evaluate.py` 的 import 语句需要更新为新模块路径

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Module split works correctly
    Tool: Bash (python)
    Preconditions: Refactored code is in place
    Steps:
      1. Run `python -c "from src.training import train_epoch, validate_epoch"` — exits 0
      2. Run `python -c "from src.visualization import plot_training_curves, plot_confusion_matrix"` — exits 0
      3. Run `python -c "from src.checkpoint import save_checkpoint, load_checkpoint"` — exits 0
      4. Run `python -c "from src.utils import train_epoch, plot_training_curves, save_checkpoint"` — exits 0 (backward compat)
    Expected Result: All imports succeed without error
    Failure Indicators: ImportError, ModuleNotFoundError
    Evidence: .sisyphus/evidence/task-1-module-split.txt

  Scenario: Existing train.py still works
    Tool: Bash (python)
    Preconditions: Module split completed
    Steps:
      1. Run `python -c "import sys; sys.path.insert(0, 'src'); from train import parse_args; print('OK')"` — exits 0
    Expected Result: train.py imports work correctly
    Failure Indicators: ImportError for training, visualization, or checkpoint
    Evidence: .sisyphus/evidence/task-1-train-import.txt
  ```

  **Commit**: YES (groups with Tasks 2, 3, 4, 6)
  - Message: `refactor: split utils.py into training, visualization, checkpoint modules`
  - Files: `src/utils.py, src/training.py, src/visualization.py, src/checkpoint.py, src/train.py, src/evaluate.py`

- [x] 2. 统一设备管理 + 消除重复代码

  **What to do**:
  - 创建 `src/device.py`，将 `get_device()` 函数放入（合并 train.py:84-94 和 evaluate.py:44-50 的两个版本，取更完整的 train.py 版本，包含GPU名称打印）
  - 在 `src/train.py` 中删除本地 `get_device()` 定义，改为 `from device import get_device`
  - 在 `src/evaluate.py` 中删除本地 `get_device()` 定义，改为 `from device import get_device`
  - 在 `src/train.py` 中删除 `main()` 内的 `from evaluate import evaluate_model` 局部import（line ~250），改为在文件顶层import或重构
  - 统一模型构建逻辑：train.py:130-135 和 evaluate.py:129-134 的 if/elif/else 创建模型逻辑重复，考虑提取到 `models/__init__.py` 的工厂函数

  **Must NOT do**:
  - 不改变 `get_device()` 的行为（auto/cuda/cpu逻辑不变）
  - 不删除GPU信息打印功能

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 消除代码重复是定位明确的快速操作
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (but logically after Task 1)
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5, 6)
  - **Blocks**: Tasks 7, 10
  - **Blocked By**: Task 1（需要先完成模块拆分再统一import）

  **References**:
  **Pattern References**:
  - `src/train.py:84-94` — `get_device()` 定义，包含GPU名称/CUDA版本打印
  - `src/evaluate.py:44-50` — `get_device()` 的简化版，缺少GPU信息打印
  - `src/train.py:250` — `from evaluate import evaluate_model` 局部import
  - `src/train.py:130-135` — 模型构建 if/elif/else
  - `src/evaluate.py:129-134` — 相同的模型构建逻辑

  **Why Each Reference Matters**:
  - train.py 的 get_device 更完整，保留这个版本
  - 模型构建重复是代码异味，需要统一入口
  - 局部import是代码异味，需移到文件顶部或工厂函数

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Device module works correctly
    Tool: Bash (python)
    Preconditions: device.py created
    Steps:
      1. Run `python -c "import sys; sys.path.insert(0, 'src'); from device import get_device; d = get_device('cpu'); print(d)"` — prints "cpu"
      2. Run `python -c "import sys; sys.path.insert(0, 'src'); from device import get_device; d = get_device('auto'); print(d)"` — prints device without error
    Expected Result: Device detection works correctly
    Failure Indicators: ImportError, RuntimeError
    Evidence: .sisyphus/evidence/task-2-device-module.txt

  Scenario: No code duplication remains
    Tool: Bash (grep)
    Preconditions: Refactoring complete
    Steps:
      1. `grep -rn "def get_device" src/` — should find exactly 1 match (in device.py)
      2. `grep -n "from device import" src/train.py src/evaluate.py` — both files import from device
    Expected Result: get_device defined in exactly 1 place, imported from 2 places
    Failure Indicators: get_device still defined in train.py or evaluate.py
    Evidence: .sisyphus/evidence/task-2-no-dup.txt
  ```

  **Commit**: YES (groups with Tasks 1, 3, 4, 6)
  - Message: `refactor: unify device management, eliminate code duplication`
  - Files: `src/device.py, src/train.py, src/evaluate.py`

- [x] 3. 补充类型注解和 docstring

  **What to do**:
  - 为所有函数添加完整的类型注解（参数和返回值）：
    - `src/data_loader.py` — `CIFAR10LocalDataset.__init__`, `_load_batches`, `__len__`, `__getitem__`, `get_transforms`, `get_data_loaders`
    - `src/training.py`（从 Task 1 拆分出来的）— `train_epoch`, `validate_epoch`
    - `src/visualization.py`（从 Task 1 拆分出来的）— `plot_training_curves`, `plot_confusion_matrix`, `plot_class_accuracy`, `print_classification_report`, `save_results_summary`
    - `src/checkpoint.py`（从 Task 1 拆分出来的）— `save_checkpoint`, `load_checkpoint`
    - `src/device.py` — `get_device`
    - `src/models/mlp.py` — `MLP.__init__`, `forward`, `create_mlp`
    - `src/models/cnn.py` — `CNN.__init__`, `forward`, `create_cnn`
    - `src/models/resnet.py` — `BasicBlock.__init__`, `forward`, `ResNet.__init__`, `_make_layer`, `forward`, `create_resnet18`
    - `src/train.py` — main 函数和辅助函数
    - `src/evaluate.py` — main 函数和辅助函数
  - 为每个函数添加 Google-style docstring（中文）
  - 为每个类添加类级 docstring

  **Must NOT do**:
  - 不改变任何函数的逻辑
  - 不添加 `typing` 模块之外的新依赖
  - 不使用过于复杂的类型注解（如嵌套3层的Dict）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 类型注解和docstring是机械性修改，无需深层设计
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Task 1 completes, since file structure changes)
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5, 6)
  - **Blocks**: Task 13 (unit tests need typed functions to test)
  - **Blocked By**: Task 1（文件拆分后才知道目标文件）

  **References**:
  **Pattern References**:
  - `src/data_loader.py:14-61` — `CIFAR10LocalDataset` 类及其方法，全无类型注解
  - `src/models/mlp.py:10-63` — `MLP` 类及工厂函数
  - `src/models/cnn.py:10-100` — `CNN` 类及工厂函数
  - `src/models/resnet.py:11-75` — `BasicBlock`, `ResNet` 类

  **Why Each Reference Matters**:
  - 这些文件是类型注解的优先目标，需要逐一添加

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Type annotations added to key functions
    Tool: Bash (python)
    Preconditions: Type annotations added
    Steps:
      1. Run `python -c "import ast, sys; sys.path.insert(0, 'src'); from training import train_epoch; sig = ast.parse('train_epoch'); print('OK')"` — exits 0
      2. Count functions without type annotations: `grep -rn "def " src/training.py src/visualization.py src/checkpoint.py src/device.py src/data_loader.py src/models/mlp.py src/models/cnn.py src/models/resnet.py | grep -v " -> " | grep -v "__" | wc -l` — should be minimal (< 3)
      3. Count functions without docstring: `python -c "import ast, inspect; ..."` — should be 0
    Expected Result: Nearly all public functions have type annotations and docstrings
    Failure Indicators: Functions still missing annotations
    Evidence: .sisyphus/evidence/task-3-type-annotations.txt
  ```

  **Commit**: YES (groups with Tasks 1, 2, 4, 6)
  - Message: `refactor: add type annotations and docstrings to all modules`
  - Files: all src/ files

- [x] 4. 完善 models/__init__.py 导出 + 统一模型构建

  **What to do**:
  - 修改 `src/models/__init__.py`，添加导出：
    ```python
    from models.mlp import MLP, create_mlp
    from models.cnn import CNN, create_cnn
    from models.resnet import ResNet, BasicBlock, create_resnet18
    
    def create_model(model_type: str, **kwargs):
        """统一模型创建工厂函数"""
        if model_type == 'mlp':
            return create_mlp(**kwargs)
        elif model_type == 'cnn':
            return create_cnn(**kwargs)
        elif model_type == 'resnet18':
            return create_resnet18(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    ```
  - 更新 `src/train.py` 中的模型创建逻辑，使用 `create_model()` 而非 if/elif/else
  - 更新 `src/evaluate.py` 中的模型创建逻辑，使用 `create_model()` 而非 if/elif/else
  - 支持从 `from models import create_model, create_mlp, create_cnn, create_resnet18` 导入

  **Must NOT do**:
  - 不修改 MLP/CNN/ResNet 的内部实现
  - 不添加作业不需要的模型变体

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的导入重构和工厂函数
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5, 6)
  - **Blocks**: Task 7 (ablation script needs create_model)
  - **Blocked By**: None

  **References**:
  **Pattern References**:
  - `src/models/__init__.py` — 当前为空文件
  - `src/train.py:130-135` — if/elif/else 模型创建逻辑
  - `src/evaluate.py:129-134` — 相同逻辑

  **Why Each Reference Matters**:
  - `__init__.py` 为空是代码质量问题，需导出模型
  - 重复的模型创建逻辑需要统一

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Model factory function works
    Tool: Bash (python)
    Preconditions: __init__.py updated
    Steps:
      1. Run `python -c "import sys; sys.path.insert(0, 'src'); from models import create_model; m = create_model('mlp'); print(type(m).__name__)"` — outputs "MLP"
      2. Run `python -c "import sys; sys.path.insert(0, 'src'); from models import create_model; m = create_model('cnn', variant='improved'); print(type(m).__name__)"` — outputs "CNN"
      3. Run `python -c "import sys; sys.path.insert(0, 'src'); from models import create_model; m = create_model('resnet18'); print(type(m).__name__)"` — outputs "ResNet"
      4. Run `python -c "import sys; sys.path.insert(0, 'src'); from models import create_model; create_model('unknown')"` — raises ValueError
    Expected Result: Factory function creates all model types correctly, rejects unknown types
    Failure Indicators: ImportError, wrong model type created, ValueError not raised for unknown
    Evidence: .sisyphus/evidence/task-4-model-factory.txt
  ```

  **Commit**: YES (groups with Tasks 1, 2, 3, 6)
  - Message: `refactor: add model exports and unified create_model factory`
  - Files: `src/models/__init__.py, src/train.py, src/evaluate.py`

- [x] 5. 数据探索可视化脚本

  **What to do**:
  - 创建 `src/data_exploration.py`，包含以下可视化：
    1. **类别分布条形图** — 训练集/验证集/测试集各类别样本数量，检查是否均衡
    2. **样本图片展示** — 每个类别随机展示8-10张图片（2x5或类似网格）
    3. **像素统计** — 每个通道的均值/标准差柱状图，验证与数据加载中的 Normalize 参数一致
    4. **数据增强效果展示** — 同一张图片经过 RandomCrop + RandomHorizontalFlip 后的效果对比（before/after）
  - 所有输出保存到 `results/data_exploration/` 目录
  - 脚本支持命令行参数 `--data_path` 和 `--output_dir`
  - 添加 `CIFAR10_CLASSES` 标签（中文标签可选）

  **Must NOT do**:
  - 不修改 `data_loader.py` 的数据加载逻辑
  - 不依赖额外的大型库（只用 matplotlib + numpy）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要数据分析和可视化设计，但不是核心算法
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (independent of code refactoring)
  - **Blocks**: Task 14 (report needs these visualizations)
  - **Blocked By**: None

  **References**:
  **Pattern References**:
  - `src/data_loader.py:64-79` — `get_transforms()` 函数，包含 Normalize 参数 (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
  - `src/data_loader.py:135-138` — `CIFAR10_CLASSES` 列表
  - `src/visualization.py` — 现有可视化模式（matplotlib风格）

  **Why Each Reference Matters**:
  - Normalize 参数需要与像素统计验证一致
  - CIFAR10_CLASSES 标签需要与图表标签一致
  - 可视化风格需要与现有图表一致（颜色主题、字体大小等）

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Data exploration runs and produces outputs
    Tool: Bash (python)
    Preconditions: Script created, CIFAR-10 data available
    Steps:
      1. Run `python src/data_exploration.py --data_path cifar-10-batches-py --output_dir results/data_exploration`
      2. Check `ls results/data_exploration/` contains at least 4 PNG files
      3. Verify each PNG is > 10KB (not empty/corrupt)
    Expected Result: 4+ PNG files generated in results/data_exploration/
    Failure Indicators: Script error, fewer than 4 PNGs, PNG files < 1KB
    Evidence: .sisyphus/evidence/task-5-data-exploration.png (screenshot of one)

  Scenario: Class distribution is balanced
    Tool: Bash (python)
    Preconditions: Script completed
    Steps:
      1. Script should output class distribution statistics to console
      2. Each class should have ~5000 training images, ~500 validation, ~1000 test
    Expected Result: Reasonable class balance in printed output
    Failure Indicators: Highly imbalanced classes (>10% deviation)
    Evidence: .sisyphus/evidence/task-5-class-balance.txt
  ```

  **Commit**: YES
  - Message: `feat: add data exploration visualization script`
  - Files: `src/data_exploration.py, results/data_exploration/`

- [x] 6. 修复验证集为分层抽样

  **What to do**:
  - 修改 `src/data_loader.py` 中的 `get_data_loaders()` 函数
  - 替换 `torch.randperm`（line 101-104）为 `sklearn.model_selection.StratifiedShuffleSplit` 或手动实现分层划分
  - 确保训练集45K和验证集5K中，每个类别的比例与原始数据集一致
  - 添加类型注解和 docstring
  - 在 `data_exploration.py` 中验证分层划分后各类别比例

  **Must NOT do**:
  - 不改变训练集和验证集的大小比例（45K/5K）
  - 不修改 `CIFAR10LocalDataset` 类的接口
  - 不引入 sklearn 作为核心依赖（仅在 data_loader 中用于分层划分）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 小而精确的代码修改
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (but logically after Task 1)
  - **Parallel Group**: Wave 1 (with Tasks 1-5)
  - **Blocks**: Task 10 (ablation experiments need proper split)
  - **Blocked By**: Task 1（文件可能在拆分后结构变化）

  **References**:
  **Pattern References**:
  - `src/data_loader.py:97-104` — 当前的随机划分逻辑（`torch.randperm` + `Subset`）
  - `src/data_loader.py:135-138` — `CIFAR10_CLASSES`

  **Why Each Reference Matters**:
  - 当前逻辑使用 `torch.randperm` 随机划分，不保证类别均衡
  - 分层抽样能提高验证集的代表性，也符合学术标准

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Stratified split produces balanced classes
    Tool: Bash (python)
    Preconditions: Stratified split implemented
    Steps:
      1. Run `python -c "import sys; sys.path.insert(0, 'src'); from data_loader import get_data_loaders; _, val_loader, _ = get_data_loaders('cifar-10-batches-py', batch_size=128)`
      2. Collect all validation labels and count per class
      3. Each class should have exactly 500 samples (or very close)
    Expected Result: Each of 10 classes has ~500 samples in validation set
    Failure Indicators: Any class deviates by more than 5%
    Evidence: .sisyphus/evidence/task-6-stratified-split.txt

  Scenario: Training still produces same-order accuracy
    Tool: Bash (python)
    Preconditions: Stratified split in place
    Steps:
      1. Run training for 1 epoch with MLP: `python src/train.py --model mlp --epochs 1 --device cpu`
      2. Verify no errors
    Expected Result: Training runs without error, accuracy is reasonable (~35-45% for 1 epoch)
    Failure Indicators: Training crashes or accuracy drops significantly
    Evidence: .sisyphus/evidence/task-6-training-works.txt
  ```

  **Commit**: YES (groups with Tasks 1-5)
  - Message: `fix: use stratified split for validation set`
  - Files: `src/data_loader.py`

- [x] 7. 消融实验框架脚本

  **What to do**:
  - 创建 `src/run_ablation.py`，支持以下消融实验配置：
    1. **数据增强消融** — 无增强 vs 仅Normalize vs RandomCrop+HFlip+Normalize
    2. **BatchNorm消融** — CNN with BN vs CNN without BN (需在CNN模型中添加BN可选参数)
    3. **Dropout消融** — Dropout=0 vs 0.3 vs 0.5（CNN/ResNet）
    4. **优化器消融** — SGD vs Adam（相同学习率）
    5. **学习率调度消融** — CosineAnnealing vs ReduceLROnPlateau vs 无调度器
  - 每个实验配置自动生成唯一的 checkpoint 和 results 子目录名（如 `ablation/no_aug/cnn/`）
  - 支持 `--dry-run` 模式（只打印配置不训练）
  - 支持 `--device` 参数
  - 自动保存实验配置JSON到结果目录
  - 所有消融实验基于CNN Improved模型（快速迭代），最后在ResNet-18上验证关键发现
  - 使用 `create_model()` 工厂函数（Task 4）

  **Must NOT do**:
  - 不修改已有模型的默认架构
  - 不删除已有checkpoint或results
  - CNN无BN变体通过参数控制，不创建新文件

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要设计实验框架、理解模型架构差异、确保实验可比性
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Wave 1)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 10 (running ablation experiments)
  - **Blocked By**: Tasks 1, 2, 4 (refactored code, unified imports)

  **References**:
  **Pattern References**:
  - `src/train.py:22-69` — argparse 参数定义（需要支持的参数：model, epochs, batch_size, lr, optimizer, scheduler）
  - `src/train.py:97-180` — 完整训练循环（需要复用或调用）
  - `src/data_loader.py:64-79` — 数据增强配置（get_transforms函数，需要为消融实验创建变体）
  - `src/models/cnn.py:10-100` — CNN模型（需要添加use_bn参数支持）
  - `src/models/__init__.py` — create_model() 工厂函数（Task 4 添加的）

  **Why Each Reference Matters**:
  - 训练参数决定了消融实验的配置空间
  - 数据增强是核心消融维度之一，需要理解现有配置
  - CNN模型需要支持BN开关才能做BN消融

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Ablation script dry-run works
    Tool: Bash (python)
    Preconditions: Script created, refactored imports available
    Steps:
      1. Run `python src/run_ablation.py --dry-run` — prints all experiment configs
      2. Verify output lists at least 5 different configurations
      3. Verify each config has: model_type, transform_type, optimizer, scheduler, unique output_dir
    Expected Result: Script prints configurations without errors, no training starts
    Failure Indicators: ImportError, missing configs, script crashes
    Evidence: .sisyphus/evidence/task-7-ablation-dryrun.txt

  Scenario: Ablation script can run single experiment
    Tool: Bash (python)
    Preconditions: Script created
    Steps:
      1. Run `python src/run_ablation.py --ablation no_aug --model mlp --epochs 1 --device cpu --output_dir results/ablation/test` — runs 1 epoch
      2. Check that `results/ablation/test/` contains config.json and training output
    Expected Result: One experiment runs to completion, config saved
    Failure Indicators: Script crashes, no output directory created
    Evidence: .sisyphus/evidence/task-7-ablation-single.txt
  ```

  **Commit**: YES
  - Message: `feat: add ablation experiment framework script`
  - Files: `src/run_ablation.py`

- [x] 8. 改进训练曲线可视化

  **What to do**:
  - 在 `src/visualization.py` 中改进 `plot_training_curves()` 函数：
    1. 添加**指数移动平均(EMA)平滑** — 默认 alpha=0.3，可选开关 `--smooth`
    2. 在 best validation accuracy 的 epoch 处添加**竖线和标注**（最佳epoch标记）
    3. 添加**train-val gap 可视化** — 训练和验证之间的差距曲线
  - 创建 `src/generate_plots.py`，一键重新生成所有改进后的可视化图：
    - 3个模型的训练曲线（MLP/CNN/ResNet-18）
    - 带平滑线的版本
    - 带 best epoch 标注的版本
  - 新函数也添加类型注解和 docstring

  **Must NOT do**:
  - 不删除旧的可视化函数（保留向后兼容）
  - 不修改训练逻辑本身

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 可视化改进需要设计和数据处理
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Task 1)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 12 (report needs improved visualizations)
  - **Blocked By**: Task 1 (visualization.py is split from utils.py)

  **References**:
  **Pattern References**:
  - `src/utils.py` or new `src/visualization.py` — `plot_training_curves()` 函数，当前实现（需要增强）
  - `results/resnet18_training_curves.png` — 当前的训练曲线（锯齿严重，需要平滑）
  - `results/mlp_training_curves.png`, `results/cnn_training_curves.png` — 其他模型的曲线

  **Why Each Reference Matters**:
  - 当前可视化函数是增强的基础
  - ResNet-18 曲线问题最严重（需要优先平滑）

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Improved visualization works
    Tool: Bash (python)
    Preconditions: visualization.py updated, generate_plots.py created
    Steps:
      1. Run `python src/generate_plots.py --data_dir results/ --output_dir results/improved/`
      2. Check that `results/improved/` contains at least 6 PNG files (3 models x 2 versions)
      3. Verify PNGs are > 20KB each (not empty)
    Expected Result: Improved training curve PNGs generated with smoothing and best-epoch markers
    Failure Indicators: Script crashes, PNGs missing or too small
    Evidence: .sisyphus/evidence/task-8-improved-curves.png

  Scenario: Smoothed curve shows best epoch
    Tool: Bash (python)
    Preconditions: Improved visualization generated
    Steps:
      1. Look at images in results/improved/ - verify they show:
         - Smoothed validation line (EMA)
         - Vertical line/badge at best epoch
         - Train-val gap visualization
    Expected Result: Images clearly show smoothed curves and best epoch markers
    Failure Indicators: Images look identical to original (no smoothing/no markers)
    Evidence: .sisyphus/evidence/task-8-curve-quality.txt
  ```

  **Commit**: YES
  - Message: `feat: improve visualization with smoothing and best-epoch markers`
  - Files: `src/visualization.py, src/generate_plots.py, results/improved/`

- [x] 9. 多模型对比可视化脚本

  **What to do**:
  - 在 `src/visualization.py` 中添加新的对比图函数：
    1. `plot_model_comparison()` — 在同一图上绘制3个模型的训练/验证精度曲线对比
    2. `plot_accuracy_bar_chart()` — 柱状图对比3个模型的测试准确率
    3. `plot_per_class_comparison()` — 10类准确率的分组柱状图
  - 创建 `src/generate_comparison.py`，读取 `results/*.json` 的结果数据，一键生成对比图
  - 输出保存到 `results/comparison/`

  **Must NOT do**:
  - 不重新训练已有模型
  - 不修改已有的单模型可视化

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 标准可视化代码，模式明确
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Task 1)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 14 (report needs comparison charts)
  - **Blocked By**: Task 1 (visualization module)

  **References**:
  **Pattern References**:
  - `results/mlp_results.json`, `results/cnn_results.json`, `results/resnet18_results.json` — 每个模型的准确率、precision、recall、F1、各类准确率
  - `results/*_training_curves.png` — 现有训练曲线（需要读数据重新绘制对比版）

  **Why Each Reference Matters**:
  - JSON 文件包含对比图需要的所有数据
  - 现有曲线可以作为参考风格

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Comparison plots generated
    Tool: Bash (python)
    Preconditions: Results JSON files exist
    Steps:
      1. Run `python src/generate_comparison.py --results_dir results/ --output_dir results/comparison/`
      2. Check `results/comparison/` contains at least 3 PNG files
      3. Verify each PNG is > 15KB
    Expected Result: 3+ comparison PNG files generated
    Failure Indicators: Script error, missing PNGs
    Evidence: .sisyphus/evidence/task-9-comparison.png

  Scenario: Comparison data includes all models
    Tool: Bash (python)
    Preconditions: Comparison generated
    Steps:
      1. Inspect generated bar chart — should show 3 models (MLP, CNN, ResNet-18)
      2. Inspect per-class comparison — should show all 10 classes
    Expected Result: All 3 models and 10 classes represented
    Failure Indicators: Missing models or classes
    Evidence: .sisyphus/evidence/task-9-data-check.txt
  ```

  **Commit**: YES
  - Message: `feat: add multi-model comparison visualization`
  - Files: `src/visualization.py, src/generate_comparison.py, results/comparison/`

- [ ] 10. 运行消融实验（数据增强 + BatchNorm + Dropout + 优化器）

  **What to do**:
  - 使用 Task 7 的 `run_ablation.py` 运行以下消融实验（需要GPU）：
    1. **数据增强消融** — CNN with augmentation vs CNN without augmentation (仅Normalize)
    2. **BatchNorm消融** — CNN with BN vs CNN without BN（修改CNN模型添加use_bn参数）
    3. **Dropout消融** — CNN with dropout=0.3 vs dropout=0 vs dropout=0.5
    4. **优化器消融** — SGD+CosineAnnealing vs Adam+CosineAnnealing（CNN）
  - 每个实验训练足够epoch（建议50-100 epoch，快速收敛即可的对比）
  - 保存结果到 `results/ablation/` 对应子目录
  - 收集所有结果到 `results/ablation/summary.json`

  **Must NOT do**:
  - 不重新训练已有的最佳模型（ResNet-18 95.16%不变）
  - 不跑过长时间的实验（每个消融最多100 epoch）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 实验设计需要理解模型行为，结果分析需要专业知识
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs Task 7 completed + GPU time)
  - **Parallel Group**: Wave 2 (sequential after Task 7)
  - **Blocks**: Task 12 (results need to be compiled)
  - **Blocked By**: Tasks 2, 6, 7

  **References**:
  **Pattern References**:
  - `src/run_ablation.py` — Task 7创建的消融实验脚本
  - `src/models/cnn.py:10-100` — CNN模型（可能需要修改添加use_bn参数）
  - `src/data_loader.py:64-79` — get_transforms()（需要为no-aug实验创建变体）
  - `results/cnn_results.json` — CNN baseline结果（83.66%，用于对比）

  **Why Each Reference Matters**:
  - 消融脚本定义了如何运行实验
  - CNN baseline是所有消融的对照组
  - 数据增强函数需要修改以支持no-aug变体

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Ablation experiments produce valid results
    Tool: Bash (python)
    Preconditions: CUDA available, ablation script runs
    Steps:
      1. Run ablation experiments (on GPU): `python src/run_ablation.py --device cuda --output_dir results/ablation`
      2. Check `results/ablation/summary.json` exists and contains results for all 4+ ablation axes
      3. Each result entry should have: experiment_name, model, test_accuracy, config
    Expected Result: summary.json has entries for all ablation experiments with accuracy values
    Failure Indicators: Missing experiments, no summary.json, accuracy values unreasonable (<30%)
    Evidence: .sisyphus/evidence/task-10-ablation-results.json

  Scenario: Ablation shows expected patterns
    Tool: Bash (python)
    Preconditions: Ablation results available
    Steps:
      1. Verify: augmentation experiment → no_aug accuracy < with_aug accuracy (expect 3-5% gap)
      2. Verify: BN experiment → no BN accuracy < with BN accuracy (expect 5-7% gap)
      3. Verify: optimizer experiment → SGD and Adam results are comparable (within 3%)
    Expected Result: Ablation results show clear trends matching known ML patterns
    Failure Indicators: Counter-intuitive results (no improvement from augmentation or BN)
    Evidence: .sisyphus/evidence/task-10-ablation-analysis.txt
  ```

  **Commit**: YES
  - Message: `feat: run ablation experiments and save results`
  - Files: `results/ablation/`

---

## Wave 3: Experiments (depends on Wave 1-2)

- [x] 11. 超参数对比实验脚本与运行

  **What to do**:
  - 创建 `src/run_experiments.py`，支持以下超参数对比（基于ResNet-18，30-50 epochs快速对比）：
    1. **学习率对比** — lr=0.01, 0.05, 0.1, 0.2（SGD + CosineAnnealing）
    2. **优化器对比** — SGD(momentum=0.9) vs Adam vs SGD(no momentum)
    3. **权重衰减对比** — weight_decay=0 vs 5e-4 vs 1e-3
  - 自动保存每组实验的 config + results + training_curves 到 `results/hyperparams/`
  - 支持 `--dry-run` 模式
  - 支持 `--device` 参数
  - 收集现有模型结果为基线数据（MLP 55.57%, CNN 83.66%, ResNet-18 95.16%）

  **Must NOT do**:
  - 不重新训练完整的200 epoch实验
  - 不修改已有模型架构

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 12, 13 if GPU available)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 14
  - **Blocked By**: Tasks 4, 5

  **References**:
  - `src/train.py:22-69` — argparse 参数
  - `src/train.py:143-164` — 训练逻辑
  - `results/resnet18_results.json` — ResNet-18 baseline 95.16%

  **Acceptance Criteria**:
  - `python src/run_experiments.py --dry-run` 列出所有实验配置
  - `results/hyperparams/` 包含至少9个实验配置的结果

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Hyperparameter script dry-run works
    Tool: Bash (python)
    Steps:
      1. Run `python src/run_experiments.py --dry-run`
      2. Verify output lists all planned experiments with configs
    Expected Result: Lists lr, optimizer, weight_decay experiments with configs
    Evidence: .sisyphus/evidence/task-11-hyperparam-dryrun.txt
  ```

  **Commit**: YES
  - Message: `feat: add hyperparameter comparison experiments`
  - Files: `src/run_experiments.py, results/hyperparams/`

- [ ] 12. ResNet-18 增强训练（Label Smoothing + Warmup）

  **What to do**:
  - 在 `src/train.py` 中添加 `--label_smoothing` 参数（默认0.0）
  - 修改 `nn.CrossEntropyLoss()` 为 `nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)`
  - 添加学习率 warmup 支持：`--warmup_epochs` 参数（默认0）
  - 实现 warmup scheduler：前 warmup_epochs 个 epoch 线性增加 lr
  - 运行 ResNet-18 增强训练：`python src/train.py --model resnet18 --epochs 200 --lr 0.1 --label_smoothing 0.1 --warmup_epochs 5 --device cuda`
  - 目标：测试准确率 > 95.16%
  - 保存结果到 `results/resnet18_enhanced/`

  **Must NOT do**:
  - 不修改 ResNet-18 架构
  - 不删除已有的 ResNet-18 baseline 结果
  - 不使用需要额外安装的库

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 11 if multiple GPUs)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 14
  - **Blocked By**: Tasks 5

  **References**:
  - `src/train.py:143-151` — 当前 CrossEntropyLoss 定义
  - `src/train.py:153-164` — 当前 scheduler 逻辑
  - `results/resnet18_results.json` — ResNet-18 baseline 95.16%

  **Acceptance Criteria**:
  - `python src/train.py --help` 包含 `label_smoothing` 和 `warmup_epochs` 参数
  - 增强训练结果 > 95.16%

  **Commit**: YES
  - Message: `feat: add label smoothing and warmup for ResNet-18 enhanced training`
  - Files: `src/train.py, results/resnet18_enhanced/`

- [x] 13. 单元测试基础设施

  **What to do**:
  - 创建 `tests/` 目录和配置：
    - `tests/conftest.py` — pytest fixtures
    - `pytest.ini` — pytest 配置
  - 创建测试文件：
    - `tests/test_models.py` — 测试3个模型前向传播
    - `tests/test_data_loader.py` — 测试数据加载、分层抽样
    - `tests/test_checkpoint.py` — 测试保存/加载检查点
    - `tests/test_device.py` — 测试 get_device
  - 所有测试可在CPU上 < 30秒完成

  **Must NOT do**:
  - 不写需要GPU的集成测试
  - 不引入 pytest 以外的测试依赖

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: F2 (code quality review)
  - **Blocked By**: Tasks 3 (type annotations needed)

  **References**:
  - `src/models/mlp.py:10-63` — MLP 类
  - `src/models/cnn.py:10-100` — CNN 类
  - `src/models/resnet.py:11-75` — ResNet 类

  **Acceptance Criteria**:
  - `pytest tests/ -v` 全部通过
  - 测试数 >= 10个

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: All unit tests pass
    Tool: Bash (python)
    Steps:
      1. Run `pytest tests/ -v`
      2. Verify all tests pass, count >= 10
    Expected Result: All tests pass on CPU in < 30 seconds
    Evidence: .sisyphus/evidence/task-13-test-results.txt
  ```

  **Commit**: YES
  - Message: `test: add unit tests for core modules`
  - Files: `tests/conftest.py, tests/test_models.py, tests/test_data_loader.py, tests/test_checkpoint.py, tests/test_device.py, pytest.ini`

---

## Wave 4: Results Integration

- [x] 14. 生成所有改进后的可视化图 + 结果汇总表

  **What to do**:
  - 使用改进后的可视化工具重新生成所有训练曲线图（平滑版+best epoch标记版）
  - 生成模型对比图（3模型 training/validation loss/accuracy 对比）
  - 生成消融实验对比柱状图
  - 生成超参数对比热力图/折线图
  - 创建 `src/generate_summary.py`，生成 Markdown 和 LaTeX 格式的实验结果表格
  - 所有输出保存到 `results/final/`

  **Must NOT do**:
  - 不手工制图
  - 不包含不存在的实验数据

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 15
  - **Blocked By**: Tasks 5, 8, 9, 10, 11, 12

  **References**:
  - `results/mlp_results.json`, `results/cnn_results.json`, `results/resnet18_results.json` — 模型结果
  - `results/ablation/summary.json` — 消融实验结果（Task 10）
  - `results/hyperparams/` — 超参数实验结果（Task 11）

  **Acceptance Criteria**:
  - `results/final/` 包含至少8个PNG文件和2个表格文件
  - 对比表包含至少3个模型行

  **Commit**: YES
  - Message: `feat: generate final visualizations and summary tables`
  - Files: `src/generate_summary.py, results/final/`

---

## Wave 5: Report

- [x] 15. 撰写完整PDF报告

  **What to do**:
  - 创建 `reports/` 目录
  - 撰写完整中文PDF报告，包含作业要求的5个部分：
    1. **问题定义与理解** — CIFAR-10 数据集、任务定义、评估指标
    2. **数据分析及处理** — 类别分布、预处理、数据增强、分层抽样
    3. **模型构建** — MLP/CNN/ResNet架构、训练策略、参数量对比
    4. **实验结果与分析** — 模型对比、消融实验、超参数实验、混淆矩阵、训练曲线分析、过拟合讨论
    5. **结论与可能改进** — 关键结论、改进方向
  - 所有数据必须来自真实实验结果文件
  - 10-15页（含图表）
  - 使用 Jupyter Notebook → PDF 或 LaTeX → PDF

  **Must NOT do**:
  - 不编造实验数据
  - 不使用模糊的AI描述（每个数据点必须有真实来源）
  - 不超过15页

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 5, 8, 9, 10, 11, 12, 14

  **References**:
  - `results/data_exploration/` — 数据探索图（Task 5）
  - `results/final/` — 整合后的可视化（Task 14）
  - `results/ablation/summary.json` — 消融结果（Task 10）
  - `results/hyperparams/` — 超参数结果（Task 11）
  - 机器学习导论2026大作业.pdf — 作业要求文档

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: PDF报告包含5个必要部分
    Tool: Bash (python)
    Steps:
      1. Check `ls reports/report.pdf` 存在
      2. 提取PDF文本，确认包含：问题定义、数据分析、模型构建、实验结果、结论
    Expected Result: PDF存在且包含5个必要部分，10-15页
    Evidence: .sisyphus/evidence/task-15-report-check.txt

  Scenario: 报告数据与实验结果一致
    Tool: Bash (python)
    Steps:
      1. 从报告提取准确率数字
      2. 与 results/ 中的 JSON 文件对比
    Expected Result: 所有数字与实际结果一致
    Evidence: .sisyphus/evidence/task-15-data-consistency.txt
  ```

  **Commit**: YES
  - Message: `docs: add complete PDF report`
  - Files: `reports/report.pdf`

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest tests/ -v`. Run `python -c "import src"` to verify imports. Review all changed files for: missing type hints, functions without docstrings, code duplication >3 lines, empty __init__.py, mixed responsibilities in modules. Check AI slop: excessive comments, over-abstraction.
  Output: `Tests [N pass/N fail] | Imports [OK/BROKEN] | Type Hints [N/N] | Docstrings [N/N] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Run data exploration script, verify PNG outputs. Run ablation experiments (1-2 quick ones), verify results JSON. Run visualization scripts, verify improved plots. Check report PDF exists and contains 5 sections. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scripts [N/N pass] | Visualizations [N/N] | Experiments [N/N] | Report [YES/NO] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `refactor: modularize utils, add type hints, fix duplication` - src/utils.py, src/training.py, src/visualization.py, src/checkpoint.py, src/device.py, src/models/__init__.py, src/data_loader.py, src/train.py, src/evaluate.py
- **Task 5**: `feat: add data exploration visualization script` - src/data_exploration.py, results/data_exploration/
- **Wave 2**: `feat: add ablation experiment framework and improved visualizations` - src/run_ablation.py, src/visualization.py updates, results/ablation/
- **Wave 3**: `feat: add hyperparameter comparison and unit tests` - src/run_hyperparam_search.py, tests/
- **Task 14**: `docs: add complete PDF report` - report.pdf

---

## Success Criteria

### Grading Target (92+ / 100)

| Dimension | Target Score | Current | Required Improvement |
|-----------|-------------|---------|---------------------|
| 代码完成情况 | 23-25 | 20-22 | 数据探索、消融实验、超参数对比、分层抽样 |
| 代码规范性 | 23-25 | 18-22 | 类型注解、消除重复、模块化、单元测试 |
| 实验效果 | 23-25 | 18-22 | 消融分析、超参数对比、可视化改进 |
| 报告规范性 | 23-25 | 0 | 完整5部分PDF |

### Verification Commands
```bash
# 代码质量
pytest tests/ -v                    # Expected: all pass
python -c "from src.models import create_mlp, create_cnn, create_resnet18"  # Expected: no error

# 数据探索
python src/data_exploration.py      # Expected: PNG files in results/data_exploration/

# 消融实验
python src/run_ablation.py --device cuda  # Expected: results in results/ablation/

# 可视化改进
python src/generate_plots.py         # Expected: improved PNG files in results/

# 报告
ls report.pdf                       # Expected: file exists
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Report PDF has all 5 sections
---

