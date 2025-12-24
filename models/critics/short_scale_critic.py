import torch
from .base_critic import BaseTrajectoryCritic


class ShortScaleCritic(BaseTrajectoryCritic):
    """
    Short-horizon critic (e.g., first 8–10 steps)
    Focus: smoothness, acceleration, jitter
    """

    def __init__(
        self,
        horizon: int = 10,
        input_dim: int = 2,
        hidden_dim: int = 128,
    ):
        self.horizon = horizon
        super().__init__(
            input_dim=horizon * input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """
        traj: [B, T, 2]
        """
        traj = traj[:, : self.horizon]
        return super().forward(traj)
