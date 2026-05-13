"""
Tests for device selection utility.
"""

import torch

from device import get_device


def test_get_device_cpu():
    """Explicit 'cpu' returns a CPU device."""
    dev = get_device('cpu')
    assert isinstance(dev, torch.device)
    assert dev.type == 'cpu'


def test_get_device_auto():
    """'auto' returns either cuda or cpu depending on availability."""
    dev = get_device('auto')
    assert isinstance(dev, torch.device)
    assert dev.type in ('cuda', 'cpu')
