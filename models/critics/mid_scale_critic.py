import torch
from .base_critic import BaseTrajectoryCritic


class MidScaleCritic(BaseTrajectoryCritic):
    """Focuses on the first 30 steps (0-3 sec). Target: Lane following/Flow."""
    def __init__(self, horizon=30, **kwargs):
        super().__init__(input_dim=4, hidden_dim=256, num_layers=3)
        self.horizon = horizon

    def forward(self, traj: torch.Tensor):
        return super().forward(traj[:, :self.horizon])