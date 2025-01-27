from collections import namedtuple

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from complexity import Complexity
from vulnerability import Vulnerability

Anomaly_iResult = namedtuple("Anomaly_iResult", ["score"])

class AnomalyScore_i:
    def __init__(self, model: nn.Module):
        """
        Args:
            model: The model to use for the anomaly score for individual images. The model's forward function should return features.
        """
        super().__init__()
        self.model = model
        self.complexity = Complexity(model)
        self.vulnerability = Vulnerability(model)

    def __call__(
        self, images_generated: torch.Tensor, batch_size=8
    ) -> Anomaly_iResult:
        assert len(images_generated.size()) == 4  # (B, C, H, W)

        gen_loader = DataLoader(images_generated, batch_size=batch_size, shuffle=False)

        complexity = []
        vulnerability = []

        for images in tqdm.tqdm(gen_loader):
            complexity.append(self.complexity(images))
            vulnerability.append(self.vulnerability(images))

        complexity = torch.cat(complexity, dim=0)
        vulnerability = torch.cat(vulnerability, dim=0)

        AS_i = torch.div(vulnerability, complexity)
        return Anomaly_iResult(value=AS_i)