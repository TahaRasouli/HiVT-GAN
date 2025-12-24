import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class BaseTrajectoryCritic(nn.Module):
    """
    Base critic for trajectory realism.
    Input: trajectories of shape [B, T, 2] or [B, T, D]
    Output: scalar realism score per trajectory
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        dim = input_dim

        for _ in range(num_layers):
            layers.append(
                spectral_norm(nn.Linear(dim, hidden_dim))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.out = spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """
        traj: [B, T, D]
        returns: [B]
        """
        B, T, D = traj.shape
        x = traj.reshape(B, T * D)
        x = self.mlp(x)
        return self.out(x).squeeze(-1)
