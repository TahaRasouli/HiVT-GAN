import torch
from .base_critic import BaseTrajectoryCritic


class LongScaleCritic(BaseTrajectoryCritic):
    """
    Full-horizon critic
    Focus: goal realism, intent consistency
    """

    def __init__(
        self,
        horizon: int,
        input_dim: int = 2,
        hidden_dim: int = 256,
    ):
        self.horizon = horizon
        super().__init__(
            input_dim=horizon * input_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        traj = traj[:, : self.horizon]
        return super().forward(traj)
