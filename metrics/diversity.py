import torch
from torchmetrics import Metric


class EndpointDiversity(Metric):
    """
    Mean pairwise distance between final endpoints across modes.
    """

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, trajs: torch.Tensor):
        """
        trajs: [F, B, T, 2]  (multi-modal predictions)
        """
        if trajs.size(0) < 2:
            return

        endpoints = trajs[:, :, -1]  # [F, B, 2]
        F = endpoints.size(0)

        dist = 0.0
        pairs = 0
        for i in range(F):
            for j in range(i + 1, F):
                dist += torch.norm(endpoints[i] - endpoints[j], dim=-1).mean()
                pairs += 1

        self.sum += dist / pairs
        self.count += 1

    def compute(self):
        return self.sum / self.count
