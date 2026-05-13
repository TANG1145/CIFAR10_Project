"""
工具函数模块 — 兼容性桥

此模块保留向后兼容性，所有函数已迁移至独立子模块：
  - src/training.py   → train_epoch(), validate_epoch()
  - src/checkpoint.py  → save_checkpoint(), load_checkpoint()
  - src/visualization.py → plot_training_curves(), plot_confusion_matrix(),
                           plot_class_accuracy(), save_results_summary(),
                           print_classification_report()

新代码请直接从对应子模块导入。
"""

try:
    # 当 src/ 在 sys.path 中时（正常脚本运行）
    from training import train_epoch, validate_epoch
    from checkpoint import save_checkpoint, load_checkpoint
    from visualization import (
        plot_training_curves,
        plot_confusion_matrix,
        plot_class_accuracy,
        save_results_summary,
        print_classification_report,
    )
except ModuleNotFoundError:
    # 当通过包导入时（from src.utils import ...）
    from .training import train_epoch, validate_epoch
    from .checkpoint import save_checkpoint, load_checkpoint
    from .visualization import (  # type: ignore
        plot_training_curves,
        plot_confusion_matrix,
        plot_class_accuracy,
        save_results_summary,
        print_classification_report,
    )
