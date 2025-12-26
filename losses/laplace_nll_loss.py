import torch
import torch.nn as nn

class LaplaceNLLLoss(nn.Module):
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # pred: [N, 4] -> loc: [N, 2], log_scale: [N, 2]
        loc, log_scale = pred.chunk(2, dim=-1)

        # 1. Clamp log_scale to prevent exp(log_scale) from exploding or vanishing
        # This keeps the gradient flow alive while maintaining stability.
        log_scale = torch.clamp(log_scale, min=-10.0, max=10.0)
        
        # 2. Convert log_scale to scale
        scale = torch.exp(log_scale)

        # 3. Calculate NLL: log(2 * scale) + |target - loc| / scale
        # Note: log(2 * scale) is equivalent to log(2) + log_scale
        nll = log_scale + torch.log(torch.tensor(2.0, device=pred.device)) + torch.abs(target - loc) / (scale + self.eps)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f'{self.reduction} is not a valid value for reduction')