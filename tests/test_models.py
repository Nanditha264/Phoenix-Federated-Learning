"""
Unit Tests for PathMNIST CNN Architecture.
"""

import torch
import pytest
from fl_benchmark.models.cnn import FedAvgCNN
from fl_benchmark.models.model_factory import create_model


def test_fedavg_cnn_parameter_count():
    """Verify exact parameter count matches audited 583,113 weights."""
    model = FedAvgCNN(in_channels=3, num_classes=9, dim=1024)
    param_info = model.count_parameters()

    assert param_info["total_parameters"] == 583113
    assert param_info["trainable_parameters"] == 583113


def test_forward_output_shape():
    """Verify forward computation shape [B, 9]."""
    model = create_model("FedAvgCNN", in_channels=3, num_classes=9, fc_hidden=1024)
    batch_size = 16
    x = torch.randn(batch_size, 3, 28, 28)
    logits = model(x)

    assert logits.shape == (batch_size, 9)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_backward_gradient_flow():
    """Verify gradient computation across all layers."""
    model = FedAvgCNN(in_channels=3, num_classes=9, dim=1024)
    x = torch.randn(8, 3, 28, 28)
    y = torch.randint(0, 9, (8,))

    criterion = torch.nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
        assert torch.abs(param.grad).sum() > 0, f"Zero gradient in {name}"


def test_invalid_input_shape_rejection():
    """Verify rejection of malformed input tensors."""
    model = FedAvgCNN(in_channels=3, num_classes=9)
    # Grayscale image (1 channel instead of 3)
    with pytest.raises(ValueError):
        model(torch.randn(4, 1, 28, 28))

    # Incorrect spatial dimensions (32x32 instead of 28x28)
    with pytest.raises(ValueError):
        model(torch.randn(4, 3, 32, 32))
