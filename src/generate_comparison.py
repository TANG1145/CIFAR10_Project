"""
多模型对比可视化生成脚本。

读取 results/*.json 的结果数据，生成对比图并保存到指定目录。
支持训练历史曲线对比（若 checkpoints 下存在 history.json）。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from visualization import (
    plot_model_comparison,
    plot_accuracy_bar_chart,
    plot_per_class_comparison,
)


def load_results(results_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """读取 results 目录下的所有模型结果 JSON 和历史数据。

    Args:
        results_dir: 存放 *_results.json 的目录路径。

    Returns:
        (results, history_data) 的元组。
        results: 模型名称到结果字典的映射。
        history_data: 模型名称到训练历史字典的映射（可能为空）。
    """
    results: Dict[str, Any] = {}
    history_data: Dict[str, Any] = {}

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    for json_file in sorted(results_path.glob("*_results.json")):
        with open(json_file, encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        model_name = data.get("model", json_file.stem.replace("_results", ""))
        results[model_name] = data

        # 尝试加载训练历史
        if "history" in data and isinstance(data["history"], dict):
            history_data[model_name] = data["history"]
        else:
            # 尝试从 checkpoints/{model}/history.json 加载
            checkpoint_dir = results_path.parent / "checkpoints" / model_name
            history_file = checkpoint_dir / "history.json"
            if history_file.exists():
                with open(history_file, encoding="utf-8") as f:
                    history_data[model_name] = json.load(f)

    return results, history_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model comparison charts")
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory containing results JSON files (default: ../results/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save comparison charts (default: ../results/comparison/)",
    )
    args = parser.parse_args()

    # 默认路径相对于 src/
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)

    results_dir = args.results_dir or os.path.join(project_dir, "results")
    output_dir = args.output_dir or os.path.join(project_dir, "results", "comparison")

    os.makedirs(output_dir, exist_ok=True)

    results, history_data = load_results(results_dir)

    if len(results) < 2:
        print(f"Warning: Only {len(results)} model(s) found. Need at least 2 for meaningful comparison.")

    print(f"Found {len(results)} model(s): {', '.join(results.keys())}")

    # 1. 训练曲线对比（仅当历史数据可用时）
    if len(history_data) >= 2:
        plot_model_comparison(
            history_data,
            os.path.join(output_dir, "model_comparison_curves.png"),
        )
    else:
        print("Skipping model comparison curves: training history not available "
              "(need at least 2 models with history).")

    # 2. 测试准确率柱状图
    accuracies: Dict[str, float] = {
        name: data["test_accuracy"] for name, data in results.items()
    }
    plot_accuracy_bar_chart(
        accuracies,
        os.path.join(output_dir, "accuracy_bar_chart.png"),
    )

    # 3. 各类别准确率分组柱状图
    per_class_accs: Dict[str, Dict[str, float]] = {
        name: data["per_class_accuracy"] for name, data in results.items()
    }
    plot_per_class_comparison(
        per_class_accs,
        os.path.join(output_dir, "per_class_comparison.png"),
    )

    print(f"\nAll comparison charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
