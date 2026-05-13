"""
Pytest fixtures for CIFAR-10 project tests.
"""

import os
import tempfile

import pytest
import torch


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a sample batch of images and labels."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


@pytest.fixture
def temp_dir() -> str:
    """Return a temporary directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for testing."""
    return torch.device('cpu')
