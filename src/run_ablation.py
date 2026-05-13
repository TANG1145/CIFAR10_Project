import os
import sys
import argparse
import json
import copy
from typing import Any, Dict, List

import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_ablation_transforms
from train import run_training, set_seed


ABLATION_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "data_aug": [
        {"name": "no_aug", "aug_type": "no_aug", "description": "No augmentation (ToTensor only)"},
        {"name": "normalize_only", "aug_type": "normalize_only", "description": "ToTensor + Normalize"},
        {"name": "full_aug", "aug_type": "full_aug", "description": "RandomCrop + HFlip + Normalize"},
    ],
    "batchnorm": [
        {"name": "with_bn", "use_bn": True, "description": "CNN with BatchNorm"},
        {"name": "no_bn", "use_bn": False, "description": "CNN without BatchNorm"},
    ],
    "dropout": [
        {"name": "dropout_0", "dropout": 0.0, "description": "Dropout = 0.0"},
        {"name": "dropout_0.3", "dropout": 0.3, "description": "Dropout = 0.3"},
        {"name": "dropout_0.5", "dropout": 0.5, "description": "Dropout = 0.5"},
    ],
    "optimizer": [
        {"name": "sgd", "optimizer": "sgd", "description": "SGD optimizer"},
        {"name": "adam", "optimizer": "adam", "description": "Adam optimizer"},
    ],
    "scheduler": [
        {"name": "cosine", "scheduler": "cosine", "description": "CosineAnnealingLR"},
        {"name": "plateau", "scheduler": "plateau", "description": "ReduceLROnPlateau"},
        {"name": "none", "scheduler": "none", "description": "No scheduler"},
    ],
}

ALL_EXPERIMENT_NAMES: set = set()
for _cfgs in ABLATION_CONFIGS.values():
    for _c in _cfgs:
        ALL_EXPERIMENT_NAMES.add(_c["name"])


def get_default_args() -> argparse.Namespace:
    default_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cifar-10-batches-py"
    )
    args = argparse.Namespace(
        data_path=default_data_path,
        batch_size=128,
        num_workers=4,
        model="cnn",
        cnn_variant="improved",
        dropout=0.3,
        epochs=200,
        lr=0.1,
        weight_decay=5e-4,
        optimizer="sgd",
        scheduler="cosine",
        early_stop=30,
        device="auto",
        seed=42,
        resume=False,
        save_dir=None,
    )
    return args


def build_experiment_configs(ablation: str, base_args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs = []
    for cfg in ABLATION_CONFIGS[ablation]:
        exp_args = copy.deepcopy(base_args)
        exp_args.save_dir = os.path.join(
            base_args.output_dir, ablation, cfg["name"], base_args.model
        )
        exp_args.experiment_name = cfg["name"]
        exp_args.ablation_type = ablation
        exp_args.description = cfg["description"]

        if "aug_type" in cfg:
            exp_args.aug_type = cfg["aug_type"]
        if "use_bn" in cfg:
            exp_args.use_bn = cfg["use_bn"]
        if "dropout" in cfg:
            exp_args.dropout = cfg["dropout"]
        if "optimizer" in cfg:
            exp_args.optimizer = cfg["optimizer"]
        if "scheduler" in cfg:
            exp_args.scheduler = cfg["scheduler"]

        configs.append(vars(exp_args))
    return configs


def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    args = argparse.Namespace(**config)
    train_transform = None
    val_transform = None

    if hasattr(args, "aug_type"):
        train_transform = get_ablation_transforms(args.aug_type)
        if args.aug_type == "no_aug":
            val_transform = get_ablation_transforms("no_aug")
        elif args.aug_type == "normalize_only":
            val_transform = get_ablation_transforms("normalize_only")
        else:
            val_transform = get_ablation_transforms("normalize_only")

    history = run_training(args, train_transform=train_transform, val_transform=val_transform)
    return history


def print_config(config: Dict[str, Any]) -> None:
    print(f"  Experiment: {config['experiment_name']}")
    print(f"    Description: {config['description']}")
    print(f"    Save Dir: {config['save_dir']}")
    keys = ["aug_type", "use_bn", "dropout", "optimizer", "scheduler"]
    for k in keys:
        if k in config:
            print(f"    {k}: {config[k]}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 Ablation Study")
    all_choices = ["all"] + list(ABLATION_CONFIGS.keys()) + sorted(ALL_EXPERIMENT_NAMES)
    parser.add_argument(
        "--ablation", type=str, default="all",
        choices=all_choices,
        help="Ablation experiment to run (category or experiment name)"
    )
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output_dir", type=str, default="results/ablation",
        help="Root output directory for all ablation experiments"
    )
    parser.add_argument("--dry_run", action="store_true", help="Print configs without training")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_args = get_default_args()
    base_args.model = args.model
    base_args.epochs = args.epochs
    base_args.device = args.device
    base_args.batch_size = args.batch_size
    base_args.lr = args.lr
    base_args.num_workers = args.num_workers
    base_args.seed = args.seed
    base_args.output_dir = args.output_dir

    if args.ablation == "all":
        ablations = list(ABLATION_CONFIGS.keys())
    elif args.ablation in ABLATION_CONFIGS:
        ablations = [args.ablation]
    else:
        ablations = []
        for _cat, _cfgs in ABLATION_CONFIGS.items():
            for _c in _cfgs:
                if _c["name"] == args.ablation:
                    ablations = [_cat]
                    break
            if ablations:
                break

    all_configs: List[Dict[str, Any]] = []
    for ablation in ablations:
        for cfg in build_experiment_configs(ablation, base_args):
            if args.ablation in ALL_EXPERIMENT_NAMES and cfg["experiment_name"] != args.ablation:
                continue
            all_configs.append(cfg)

    if args.dry_run:
        print("=" * 60)
        print("ABlation Study Dry Run")
        print("=" * 60)
        print(f"Total experiments: {len(all_configs)}\n")
        for config in all_configs:
            print_config(config)
        return

    print("=" * 60)
    print("Starting Ablation Study")
    print("=" * 60)
    print(f"Total experiments: {len(all_configs)}\n")

    results_summary = []
    for i, config in enumerate(all_configs, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(all_configs)}: {config['experiment_name']}")
        print(f"Ablation: {config['ablation_type']}")
        print(f"{'='*60}")
        print_config(config)

        try:
            history = run_single_experiment(config)
            best_val_acc = max(history.get("val_acc", [0.0]))
            results_summary.append({
                "ablation": config["ablation_type"],
                "experiment": config["experiment_name"],
                "best_val_acc": best_val_acc,
                "save_dir": config["save_dir"],
            })
        except Exception as e:
            print(f"Error running experiment {config['experiment_name']}: {e}")
            results_summary.append({
                "ablation": config["ablation_type"],
                "experiment": config["experiment_name"],
                "error": str(e),
                "save_dir": config["save_dir"],
            })

    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("Ablation Study Complete")
    print("=" * 60)
    for r in results_summary:
        if "best_val_acc" in r:
            print(f"{r['ablation']:12s} / {r['experiment']:15s} -> Best Val Acc: {r['best_val_acc']:.2f}%")
        else:
            print(f"{r['ablation']:12s} / {r['experiment']:15s} -> ERROR: {r.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
