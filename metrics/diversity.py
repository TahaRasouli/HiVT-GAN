from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class EndpointDiversity(Metric):
    """
    Average pairwise distance between final positions across modes.
    Higher is better.
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(EndpointDiversity, self).__init__(
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
            pred: [F, B, T, 2] multi-modal trajectories
        """
        F, B, _, _ = pred.shape
        if F < 2:
            return

        endpoints = pred[:, :, -1]  # [F, B, 2]
        endpoints = endpoints.permute(1, 0, 2)  # [B, F, 2]

        dist = torch.cdist(endpoints, endpoints, p=2)  # [B, F, F]

        # Upper triangle without diagonal
        triu_indices = torch.triu_indices(F, F, offset=1)
        pairwise_dist = dist[:, triu_indices[0], triu_indices[1]]  # [B, F*(F-1)/2]

        self.sum += pairwise_dist.mean(dim=-1).sum()
        self.count += B

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
