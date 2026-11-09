"""
FedAvg Client Implementation.
Executes local Empirical Risk Minimization (ERM) using Stochastic Gradient Descent.
"""

from typing import Tuple
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.federated.client import FederatedClient
from src.metrics.update_geometry import compute_client_update
from src.metrics.loss import RunningAverage


class FedAvgClient(FederatedClient):
    """
    Standard Federated Averaging (FedAvg) client.
    Optimizes standard cross-entropy loss over local batches without proximal regularizers.
    """

    def train(
        self,
        global_model: nn.Module
    ) -> Tuple[nn.Module, torch.Tensor, float]:
        """
        Train local model using standard SGD.
        """
        # 1. Initialize local model with current global parameters
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        loss_meter = RunningAverage()

        # 2. Local Training Epochs
        for epoch in range(self.local_epochs):
            for images, targets in self.dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = local_model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), n=targets.size(0))

        # 3. Compute parameter displacement: Delta_k = w_k - w_global
        client_update = compute_client_update(local_model, global_model)

        return local_model, client_update, loss_meter.avg
