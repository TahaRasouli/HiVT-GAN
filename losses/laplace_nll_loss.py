import torch
import torch.nn as nn

class LaplaceNLLLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [N, 4] -> loc: [N, 2], scale: [N, 2]
        # We chunk the last dimension into location and raw scale
        loc, scale = pred.chunk(2, dim=-1)

        # 1. Laplace NLL Formula: log(2 * scale) + |target - loc| / scale
        # We use a safe scale to prevent division by zero or log(0)
        safe_scale = scale + self.eps
        
        # log(2 * scale) is the 'uncertainty' penalty
        # |target - loc| / scale is the 'accuracy' penalty
        nll = torch.log(2.0 * safe_scale) + torch.abs(target - loc) / safe_scale

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f'{self.reduction} is not a valid value for reduction')