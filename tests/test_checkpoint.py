"""
Tests for checkpoint save/load utilities.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from checkpoint import save_checkpoint, load_checkpoint
from models import create_model


def test_save_load_checkpoint(temp_dir, device):
    """Saving and loading a checkpoint restores model parameters."""
    model = create_model('mlp')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Initialize model with a deterministic seed so weights are known
    torch.manual_seed(0)
    model = create_model('mlp')
    model.to(device)

    path = os.path.join(temp_dir, "checkpoint.pth")
    epoch = 5
    best_acc = 0.85

    save_checkpoint(model, optimizer, epoch, best_acc, path)
    assert os.path.isfile(path)

    # Create a fresh model and optimizer, then load
    model2 = create_model('mlp')
    model2.to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    loaded_epoch, loaded_best_acc = load_checkpoint(model2, optimizer2, path, device)

    assert loaded_epoch == epoch
    assert loaded_best_acc == best_acc

    # Verify parameters are identical
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
