import torch
from .base_critic import BaseTrajectoryCritic


class LongScaleCritic(BaseTrajectoryCritic):
    """Focuses on the full horizon. Target: Goal reached/Intent."""
    def __init__(self, horizon=None, **kwargs):
        super().__init__(input_dim=4, hidden_dim=256, num_layers=4)

    def forward(self, traj: torch.Tensor):
        return super().forward(traj)