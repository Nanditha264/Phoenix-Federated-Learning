"""
FedProx Client Implementation.
Incorporates a proximal regularizer (mu / 2) * ||w - w_global||^2 to constrain client drift.
"""

from typing import Tuple, Optional
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.federated.client import FederatedClient
from src.metrics.update_geometry import compute_client_update
from src.metrics.loss import RunningAverage


class FedProxClient(FederatedClient):
    """
    Federated Proximal (FedProx) client.
    Constrains local updates towards global parameters via an explicit proximal term:
      Loss = CrossEntropy(logits, targets) + (mu / 2) * sum_{p} ||w_p - w_global_p||^2
    """

    def __init__(
        self,
        cid: int,
        dataset,
        mu: float = 0.01,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        local_epochs: int = 2,
        device: Optional[torch.device] = None,
        is_noisy: bool = False,
    ):
        super().__init__(
            cid=cid,
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            local_epochs=local_epochs,
            device=device,
            is_noisy=is_noisy,
        )
        self.mu = float(mu)
        assert self.mu >= 0.0, f"FedProx mu must be non-negative, got {self.mu}"

    def compute_proximal_term(
        self,
        local_model: nn.Module,
        global_ref_model: nn.Module,
    ) -> torch.Tensor:
        """
        Calculate proximal penalty: (mu / 2) * sum_{p} ||w_local - w_global||^2.
        Only sums over trainable parameters.
        """
        if self.mu == 0.0:
            return torch.tensor(0.0, device=self.device)

        prox_norm = torch.tensor(0.0, device=self.device)
        for (name_local, param_local), (name_ref, param_ref) in zip(
            local_model.named_parameters(), global_ref_model.named_parameters()
        ):
            if param_local.requires_grad:
                prox_norm = prox_norm + torch.sum((param_local - param_ref.to(self.device)) ** 2)

        return (self.mu / 2.0) * prox_norm

    def train(
        self,
        global_model: nn.Module
    ) -> Tuple[nn.Module, torch.Tensor, float]:
        """
        Train local model using SGD with proximal regularizer.
        """
        # 1. Initialize local model with current global parameters
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        # 2. Maintain immutable reference to global parameters
        global_ref = copy.deepcopy(global_model).to(self.device)
        global_ref.eval()
        for p in global_ref.parameters():
            p.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        loss_meter = RunningAverage()

        # 3. Local Training Epochs with Proximal Penalty
        for epoch in range(self.local_epochs):
            for images, targets in self.dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = local_model(images)
                task_loss = criterion(logits, targets)

                if self.mu > 0.0:
                    prox_loss = self.compute_proximal_term(local_model, global_ref)
                    total_loss = task_loss + prox_loss
                else:
                    total_loss = task_loss

                total_loss.backward()
                optimizer.step()

                loss_meter.update(task_loss.item(), n=targets.size(0))

        # 4. Compute parameter displacement: Delta_k = w_k - w_global
        client_update = compute_client_update(local_model, global_model)

        # 5. Clean up reference model from memory
        del global_ref

        return local_model, client_update, loss_meter.avg
