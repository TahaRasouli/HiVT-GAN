import torch
from torchmetrics import Metric


class SpeedViolation(Metric):
    """
    Fraction of trajectories exceeding a speed threshold.
    """

    def __init__(self, speed_limit: float = 15.0):
        super().__init__()
        self.speed_limit = speed_limit
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, traj: torch.Tensor):
        """
        traj: [B, T, 2]
        """
        vel = traj[:, 1:] - traj[:, :-1]
        speed = torch.norm(vel, dim=-1)

        self.sum += (speed > self.speed_limit).any(dim=1).float().sum()
        self.count += traj.size(0)

    def compute(self):
        return self.sum / self.count
