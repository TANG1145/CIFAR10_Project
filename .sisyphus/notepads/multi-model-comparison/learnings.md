# Learnings - Multi-Model Comparison Visualization

## Data Structure
- `results/*_results.json`: Contains `model` (str), `test_accuracy` (float, %), `per_class_accuracy` (dict, 0-1 ratios)
- Training history is NOT in results JSON — it's saved to `checkpoints/{model}/history.json` by `train.py`
- History JSON contains `train_loss`, `train_acc`, `val_loss`, `val_acc` (all lists)
- In this project, checkpoints/ directory was cleaned up so no history data exists

## Implementation Decisions
- `plot_model_comparison()` accepts `Dict[str, Dict[str, List[float]]]` — flexible, works with any number of models
- `plot_accuracy_bar_chart()` and `plot_per_class_comparison()` accept dicts keyed by model name
- Per-class accuracies are stored as 0-1 ratios in JSON; chart converts to percentages (0-100) for display
- Colors: consistent tab10-based palette shared across all 3 comparison functions
- Line styles: solid for validation, dashed for training (in plot_model_comparison)

## Generation Script Behavior
- `generate_comparison.py` tries to load history from: (1) embedded in results JSON → (2) checkpoints/{model}/history.json
- Skips `plot_model_comparison` gracefully when < 2 histories available
- Default paths relative to project root; all overridable via CLI args

## Verification Results
- Syntax: OK (python3 compile)
- 3 PNGs generated in results/comparison/:
  - accuracy_bar_chart.png (93K, 2964x1770)
  - per_class_comparison.png (143K, 4164x1763)
  - model_comparison_curves.png (476K, 4764x1764, verified with synthetic data)
