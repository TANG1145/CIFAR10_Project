"""
Tests for data loading, splitting, and transforms.
"""

import os

import numpy as np
import pytest
from torchvision import transforms

from data_loader import get_data_loaders, get_transforms


def _data_path() -> str:
    """Resolve CIFAR-10 data directory relative to project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "cifar-10-batches-py")


def test_data_loader_split():
    """Train/val/test splits have reasonable lengths."""
    data_path = _data_path()
    if not os.path.isdir(data_path):
        pytest.skip("CIFAR-10 data not available")

    train_loader, val_loader, test_loader = get_data_loaders(
        data_path=data_path, batch_size=128, num_workers=0
    )

    assert len(train_loader.dataset) > len(val_loader.dataset)
    assert len(test_loader.dataset) == 10_000


def test_stratified_split():
    """Validation set has roughly balanced classes (~500 each)."""
    data_path = _data_path()
    if not os.path.isdir(data_path):
        pytest.skip("CIFAR-10 data not available")

    _, val_loader, _ = get_data_loaders(
        data_path=data_path, batch_size=128, num_workers=0
    )

    labels = []
    for _, batch_labels in val_loader:
        labels.extend(batch_labels.tolist())
    labels = np.array(labels)

    for c in range(10):
        count = np.sum(labels == c)
        assert 450 <= count <= 550, f"Class {c} has {count} samples, expected ~500"


def test_transforms():
    """get_transforms returns a torchvision Compose object."""
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    assert isinstance(train_transform, transforms.Compose)
    assert isinstance(val_transform, transforms.Compose)
