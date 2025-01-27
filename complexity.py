from typing import Literal

import torch
from einops import rearrange
from torch import nn


class Complexity:
    def __init__(
        self,
        model: nn.Module,
        method: Literal["uniform", "gaussian"] = "gaussian",
    ):
        self.model = model
        self.method = method
        self.device = next(self.model.parameters()).device

    def __call__(
        self, image: torch.Tensor, num_steps: int = 10, eps: float = 0.01
    ) -> torch.Tensor:
        assert len(image.size()) == 4
        assert not self.model.training

        noise = self._get_noise(image.size()).to(self.device)
        steps = torch.arange(num_steps + 1).to(self.device)

        noises = steps[None, :, None, None, None] * noise[:, None, :, :, :]

        noisy_images = image[:, None, :, :, :] + eps * noises
        noisy_images = torch.clamp(noisy_images, min=0, max=1)

        noisy_images = rearrange(noisy_images, "b k c h w -> (b k) c h w")
        noisy_features = self.model.forward(noisy_images).clone().detach()
        noisy_features = rearrange(noisy_features, "(b k) d -> b k d", k=num_steps + 1)

        vecs = noisy_features[:, 1:, :] - noisy_features[:, :-1, :]
        vecs /= torch.norm(vecs, dim=2, p=2, keepdim=True)

        sims = torch.einsum("bkd,bkd->bk", vecs[:, 1:, :], vecs[:, :-1, :])

        angles = torch.arccos(sims)
        self.angles = angles

        return angles.mean(axis=1)

    def _get_noise(self, size: torch.Size):
        if self.method == "gaussian":
            noise = torch.randn(size, device=self.device)
        elif self.method == "uniform":
            noise = torch.randint(-1, 2, size, device=self.device).float()
        else:
            raise ValueError(f"Invalid method: {self.method}")

        noise /= torch.linalg.vector_norm(noise, dim=(1, 2, 3), keepdim=True)

        return noise
