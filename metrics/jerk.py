import torch
from torchmetrics import Metric


class Jerk(Metric):
    """
    Mean squared jerk (3rd derivative of position).
    """

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, traj: torch.Tensor):
        """
        traj: [B, T, 2]
        """
        if traj.size(1) < 4:
            return

        vel = traj[:, 1:] - traj[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]  # [B, T-3, 2]

        self.sum += jerk.pow(2).sum(dim=-1).mean(dim=-1).sum()
        self.count += traj.size(0)

    def compute(self):
        return self.sum / self.count
