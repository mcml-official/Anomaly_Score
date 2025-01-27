from collections import namedtuple

import ndtest
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from complexity import Complexity
from vulnerability import Vulnerability

AnomalyResult = namedtuple("AnomalyResult", ["p_value", "distance"])


class AnomalyScore:
    def __init__(self, model: nn.Module):
        """
        Args:
            model: The model to use for the anomaly score. The model's forward function should return features.
        Note:
            The ndtest module is used for performing a 2D Kolmogorov-Smirnov test between two samples.
            See https://github.com/syrte/ndtest for details.
        """
        super().__init__()
        self.model = model
        self.complexity = Complexity(model)
        self.vulnerability = Vulnerability(model)

    def __call__(
        self, images_real: torch.Tensor, images_generated: torch.Tensor, batch_size=8
    ) -> AnomalyResult:
        assert len(images_real.size()) == 4  # (B, C, H, W)
        assert len(images_generated.size()) == 4  # (B, C, H, W)
        assert images_real.size(0) == images_generated.size(0)  # B

        real_loader = DataLoader(images_real, batch_size=batch_size, shuffle=False)
        gen_loader = DataLoader(images_generated, batch_size=batch_size, shuffle=False)

        complexity_real = []
        vulnerability_real = []
        complexity_generated = []
        vulnerability_generated = []

        for images in tqdm.tqdm(real_loader):
            complexity_real.append(self.complexity(images))
            vulnerability_real.append(self.vulnerability(images))

        for images in tqdm.tqdm(gen_loader):
            complexity_generated.append(self.complexity(images))
            vulnerability_generated.append(self.vulnerability(images))

        complexity_real = torch.cat(complexity_real, dim=0)
        vulnerability_real = torch.cat(vulnerability_real, dim=0)
        complexity_generated = torch.cat(complexity_generated, dim=0)
        vulnerability_generated = torch.cat(vulnerability_generated, dim=0)

        P, D = ndtest.ks2d2s(
            complexity_real.cpu().numpy(),
            vulnerability_real.cpu().numpy(),
            complexity_generated.cpu().numpy(),
            vulnerability_generated.cpu().numpy(),
            extra=True,
        )

        return AnomalyResult(p_value=P, distance=D)