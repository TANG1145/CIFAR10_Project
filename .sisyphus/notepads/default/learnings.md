# Learnings

- Project uses `src/` as PYTHONPATH root; models import via `from models.mlp import ...` not `from src.models.mlp import ...`. pytest.ini must set `pythonpath = src`.
- CIFAR-10 data exists locally at `cifar-10-batches-py/` but is gitignored; tests should skip gracefully if data is missing (`pytest.skip`).
- `pin_memory=True` in DataLoader triggers a warning on Mac (MPS), but tests still pass.
- NumPy 2.4+ deprecation warning in `pickle.load` from data_loader.py is pre-existing and doesn't affect test results.
- All 11 tests passed on CPU in ~1.35 seconds, well under the 30-second budget.
