import torch
from torchmetrics import Metric


class ADE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # pred, target: [B, T, 2]
        self.sum += torch.norm(pred - target, p=2, dim=-1).mean(dim=-1).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
