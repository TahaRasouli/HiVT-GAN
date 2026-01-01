import torch
import torch.nn as nn


class R1Regularization(nn.Module):
    def __init__(self):
        super(R1Regularization, self).__init__()

    def forward(self, d_out: torch.Tensor, real_data: torch.Tensor) -> torch.Tensor:
        # CRITICAL FIX: retain_graph=True
        # This prevents autograd.grad from deleting the graph that d_loss needs later.
        grad_real = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,  # <--- THIS IS THE FIX
            only_inputs=True,
        )[0]
        
        grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(dim=1)
        return grad_penalty.mean()