"""
PathMNIST Convolutional Neural Network Architecture.
Matches the audited PFLlib FedAvgCNN architecture adapted for 3-channel RGB PathMNIST inputs and 9 output logits.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn


class FedAvgCNN(nn.Module):
    """
    Standard 2-layer Convolutional Neural Network for PathMNIST federated benchmarking.
    Architecture:
      Conv1: (in_channels=3, out_channels=32, k=5, s=1, p=0) -> ReLU -> MaxPool(2x2) [output 12x12]
      Conv2: (in_channels=32, out_channels=64, k=5, s=1, p=0) -> ReLU -> MaxPool(2x2) [output 4x4]
      Flatten: 64 * 4 * 4 = 1024 features
      FC1: Linear(1024, 512) -> ReLU
      FC:  Linear(512, num_classes=9)
    Total parameters for PathMNIST (3 in, 9 out): 583,113
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        dim: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=5,
                padding=0,
                stride=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=5,
                padding=0,
                stride=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.
        Args:
            x: Input tensor of shape (B, 3, 28, 28).
        Returns:
            Logits tensor of shape (B, 9).
        """
        if x.ndim != 4 or x.shape[1] != self.in_channels or x.shape[2:] != (28, 28):
            raise ValueError(
                f"Expected input shape (B, {self.in_channels}, 28, 28), got {tuple(x.shape)}"
            )

        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        logits = self.fc(out)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """
        Return total and trainable parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
        }
