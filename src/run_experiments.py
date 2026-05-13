"""
超参数对比实验脚本
支持学习率、优化器、权重衰减的网格对比实验（基于 ResNet-18）
"""

import os
import sys
import argparse
import json
import copy
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import run_training, set_seed
from visualization import plot_training_curves


BASELINE_RESULTS = {
    "mlp": {"test_acc": 55.57, "epochs": 200},
    "cnn": {"test_acc": 83.66, "epochs": 200},
    "resnet18": {"test_acc": 95.16, "epochs": 200},
}

EXPERIMENTS: List[Dict[str, Any]] = [
    {"name": "lr_0.01", "lr": 0.01, "optimizer": "sgd", "weight_decay": 5e-4,
     "scheduler": "cosine", "description": "Learning rate = 0.01 (SGD + Cosine)"},
    {"name": "lr_0.05", "lr": 0.05, "optimizer": "sgd", "weight_decay": 5e-4,
     "scheduler": "cosine", "description": "Learning rate = 0.05 (SGD + Cosine)"},
    {"name": "lr_0.1", "lr": 0.1, "optimizer": "sgd", "weight_decay": 5e-4,
     "scheduler": "cosine", "description": "Learning rate = 0.1 (SGD + Cosine)"},
    {"name": "lr_0.2", "lr": 0.2, "optimizer": "sgd", "weight_decay": 5e-4,
     "scheduler": "cosine", "description": "Learning rate = 0.2 (SGD + Cosine)"},
    {"name": "opt_sgd", "lr": 0.1, "optimizer": "sgd", "weight_decay": 5e-4,
     "scheduler": "cosine", "description": "SGD with momentum=0.9 + Cosine"},
    {"name": "opt_adam", "lr": 0.1, "optimizer": "adam", "weight_decay": 5e-4,
     "scheduler": "cosine", "description": "Adam + Cosine"},
    {"name": "opt_sgd_no_momentum", "lr": 0.1, "optimizer": "sgd",
     "weight_decay": 5e-4, "scheduler": "cosine", "momentum": 0.0,
     "description": "SGD without momentum + Cosine"},
    {"name": "wd_0", "lr": 0.1, "optimizer": "sgd", "weight_decay": 0.0,
     "scheduler": "cosine", "description": "Weight decay = 0 (SGD + Cosine)"},
    {"name": "wd_5e-4", "lr": 0.1, "optimizer": "sgd", "weight_decay": 5e-4,
     "scheduler": "cosine", "description": "Weight decay = 5e-4 (SGD + Cosine)"},
    {"name": "wd_1e-3", "lr": 0.1, "optimizer": "sgd", "weight_decay": 1e-3,
     "scheduler": "cosine", "description": "Weight decay = 1e-3 (SGD + Cosine)"},
]


def get_default_args() -> argparse.Namespace:
    default_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cifar-10-batches-py"
    )
    args = argparse.Namespace(
        data_path=default_data_path,
        batch_size=128,
        num_workers=4,
        model="resnet18",
        cnn_variant="improved",
        dropout=0.3,
        epochs=50,
        lr=0.1,
        weight_decay=5e-4,
        optimizer="sgd",
        momentum=0.9,
        scheduler="cosine",
        early_stop=30,
        device="auto",
        seed=42,
        resume=False,
        save_dir=None,
    )
    return args


def build_experiment_args(
    exp_cfg: Dict[str, Any],
    base_args: argparse.Namespace,
    output_dir: str,
) -> argparse.Namespace:
    exp_args = copy.deepcopy(base_args)
    exp_args.save_dir = os.path.join(output_dir, exp_cfg["name"])
    exp_args.experiment_name = exp_cfg["name"]
    exp_args.description = exp_cfg["description"]

    for key in ["lr", "optimizer", "weight_decay", "scheduler", "momentum"]:
        if key in exp_cfg:
            setattr(exp_args, key, exp_cfg[key])

    return exp_args


def print_experiment_config(args: argparse.Namespace) -> None:
    print(f"  Experiment: {args.experiment_name}")
    print(f"    Description: {args.description}")
    print(f"    Save Dir: {args.save_dir}")
    print(f"    Model: {args.model}, Epochs: {args.epochs}")
    print(f"    LR: {args.lr}, Optimizer: {args.optimizer}")
    print(f"    Weight Decay: {args.weight_decay}, Scheduler: {args.scheduler}")
    if hasattr(args, "momentum") and args.optimizer == "sgd":
        print(f"    Momentum: {args.momentum}")
    print()


def run_single_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    history = run_training(args)

    curves_path = os.path.join(args.save_dir, "training_curves.png")
    plot_training_curves(history, curves_path)

    best_val_acc = max(history.get("val_acc", [0.0]))
    final_train_acc = history["train_acc"][-1] if history.get("train_acc") else 0.0
    final_val_acc = history["val_acc"][-1] if history.get("val_acc") else 0.0

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 Hyperparameter Comparison")
    parser.add_argument(
        "--model", type=str, default="resnet18",
        choices=["mlp", "cnn", "resnet18"],
        help="Model type (default: resnet18)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50 for quick comparison)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: auto/cuda/cpu",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/hyperparams",
        help="Root output directory for all experiments",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print experiment configs without training",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_args = get_default_args()
    base_args.model = args.model
    base_args.epochs = args.epochs
    base_args.device = args.device
    base_args.seed = args.seed

    # 构建所有实验参数
    all_exp_args: List[argparse.Namespace] = []
    for exp_cfg in EXPERIMENTS:
        exp_args = build_experiment_args(exp_cfg, base_args, args.output_dir)
        all_exp_args.append(exp_args)

    # Dry run 模式
    if args.dry_run:
        print("=" * 60)
        print("Hyperparameter Comparison Dry Run")
        print("=" * 60)
        print(f"Total experiments: {len(all_exp_args)}\n")
        for exp_args in all_exp_args:
            print_experiment_config(exp_args)
        return

    # 正式训练
    print("=" * 60)
    print("Starting Hyperparameter Comparison")
    print("=" * 60)
    print(f"Model: {args.model}, Epochs: {args.epochs}, Device: {args.device}")
    print(f"Total experiments: {len(all_exp_args)}\n")

    results_summary: List[Dict[str, Any]] = []
    for i, exp_args in enumerate(all_exp_args, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(all_exp_args)}: {exp_args.experiment_name}")
        print(f"{'='*60}")
        print_experiment_config(exp_args)

        try:
            result = run_single_experiment(exp_args)
            results_summary.append({
                "experiment": exp_args.experiment_name,
                "description": exp_args.description,
                "config": {
                    "model": exp_args.model,
                    "epochs": exp_args.epochs,
                    "lr": exp_args.lr,
                    "optimizer": exp_args.optimizer,
                    "weight_decay": exp_args.weight_decay,
                    "scheduler": exp_args.scheduler,
                    "momentum": getattr(exp_args, "momentum", 0.9),
                },
                "best_val_acc": result["best_val_acc"],
                "final_train_acc": result["final_train_acc"],
                "final_val_acc": result["final_val_acc"],
                "save_dir": exp_args.save_dir,
            })
        except Exception as e:
            print(f"Error running experiment {exp_args.experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                "experiment": exp_args.experiment_name,
                "description": exp_args.description,
                "config": {
                    "model": exp_args.model,
                    "epochs": exp_args.epochs,
                    "lr": exp_args.lr,
                    "optimizer": exp_args.optimizer,
                    "weight_decay": exp_args.weight_decay,
                    "scheduler": exp_args.scheduler,
                    "momentum": getattr(exp_args, "momentum", 0.9),
                },
                "error": str(e),
                "save_dir": exp_args.save_dir,
            })

    # 保存汇总结果
    summary = {
        "baseline_results": BASELINE_RESULTS,
        "experiments": results_summary,
        "model": args.model,
        "epochs": args.epochs,
        "total_experiments": len(all_exp_args),
        "completed": sum(1 for r in results_summary if "error" not in r),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # 打印结果汇总表
    print("\n" + "=" * 60)
    print("Hyperparameter Comparison Complete")
    print("=" * 60)
    print(f"{'Experiment':<25s} {'Best Val Acc':>12s}")
    print("-" * 60)
    for r in results_summary:
        if "best_val_acc" in r:
            print(f"{r['experiment']:<25s} {r['best_val_acc']:>11.2f}%")
        else:
            print(f"{r['experiment']:<25s} {'ERROR':>12s}")
    print("-" * 60)
    for model_name, baseline in BASELINE_RESULTS.items():
        print(f"{'[Baseline] ' + model_name:<25s} {baseline['test_acc']:>11.2f}%")


if __name__ == "__main__":
    main()
