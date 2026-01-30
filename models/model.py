import torch
from torch import nn

from .cnn import CNNFeatureExtractor
from .mlp import MLPRegressionHead


class CNNMLPRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.mlp = MLPRegressionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.cnn(x)
        return self.mlp(embedding)

