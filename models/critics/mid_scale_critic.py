import torch
from .base_critic import BaseTrajectoryCritic


class MidScaleCritic(BaseTrajectoryCritic):
    """
    Mid-horizon critic (e.g., first 20 steps)
    Focus: interaction realism
    """

    def __init__(
        self,
        horizon: int = 20,
        input_dim: int = 2,
        hidden_dim: int = 256,
    ):
        self.horizon = horizon
        super().__init__(
            input_dim=horizon * input_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        traj = traj[:, : self.horizon]
        return super().forward(traj)
