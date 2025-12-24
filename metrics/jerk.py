from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class Jerk(Metric):
    """
    Average jerk (third-order finite difference) over trajectories.
    Lower is better.
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(Jerk, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor) -> None:
        """
        Args:
            pred: [B, T, 2] predicted trajectories
        """
        if pred.size(1) < 4:
            return

        jerk = (
            pred[:, 3:]
            - 3 * pred[:, 2:-1]
            + 3 * pred[:, 1:-2]
            - pred[:, :-3]
        )  # [B, T-3, 2]

        jerk_norm = torch.norm(jerk, p=2, dim=-1).mean(dim=-1)  # [B]
        self.sum += jerk_norm.sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
