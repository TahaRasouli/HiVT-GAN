from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class SpeedViolation(Metric):
    """
    Fraction of timesteps violating speed or acceleration limits.
    Lower is better.
    """

    def __init__(
        self,
        max_speed: float = 10.0,        # m/s (≈ 36 km/h)
        max_accel: float = 5.0,         # m/s^2
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(SpeedViolation, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.max_speed = max_speed
        self.max_accel = max_accel

        self.add_state("violations", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor) -> None:
        """
        Args:
            pred: [B, T, 2] predicted trajectories
        """
        if pred.size(1) < 3:
            return

        velocity = pred[:, 1:] - pred[:, :-1]           # [B, T-1, 2]
        speed = torch.norm(velocity, p=2, dim=-1)       # [B, T-1]

        accel = velocity[:, 1:] - velocity[:, :-1]      # [B, T-2, 2]
        accel_norm = torch.norm(accel, p=2, dim=-1)     # [B, T-2]

        speed_violation = speed > self.max_speed
        accel_violation = accel_norm > self.max_accel

        total_violations = speed_violation.sum() + accel_violation.sum()
        total_steps = speed.numel() + accel_norm.numel()

        self.violations += total_violations.float()
        self.count += total_steps

    def compute(self) -> torch.Tensor:
        return self.violations / self.count
