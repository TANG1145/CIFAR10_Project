"""
Tests for model definitions and factory functions.
"""

import torch

from models import create_model
from models.mlp import create_mlp
from models.cnn import create_cnn
from models.resnet import create_resnet18


def test_mlp_forward(sample_batch):
    """MLP forward pass produces correct output shape."""
    model = create_mlp(dropout_rate=0.3)
    images, _ = sample_batch
    out = model(images)
    assert out.shape == (4, 10)


def test_cnn_forward(sample_batch):
    """CNN forward pass produces correct output shape."""
    model = create_cnn(variant='improved', dropout_rate=0.3)
    images, _ = sample_batch
    out = model(images)
    assert out.shape == (4, 10)


def test_resnet18_forward(sample_batch):
    """ResNet-18 forward pass produces correct output shape."""
    model = create_resnet18()
    images, _ = sample_batch
    out = model(images)
    assert out.shape == (4, 10)


def test_mlp_param_count():
    """MLP has a positive number of parameters."""
    model = create_mlp()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0


def test_model_factory():
    """create_model factory returns valid models for all supported types."""
    for model_type in ['mlp', 'cnn', 'resnet18']:
        model = create_model(model_type)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)
