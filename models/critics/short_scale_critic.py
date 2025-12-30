import torch
from .base_critic import BaseTrajectoryCritic


class ShortScaleCritic(BaseTrajectoryCritic):
    """Focuses on the first 10 steps (0-1 sec). Target: Acceleration spikes/Jerk."""
    def __init__(self, horizon=10, **kwargs):
        super().__init__(input_dim=4, hidden_dim=128, num_layers=2)
        self.horizon = horizon

    def forward(self, traj: torch.Tensor):
        return super().forward(traj[:, :self.horizon])
