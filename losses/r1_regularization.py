import torch
import torch.nn as nn


class R1Regularization(nn.Module):
    """
    R1 regularization on real samples.
    Penalizes ||∇_x D(x)||^2 for real trajectories.
    """

    def __init__(self):
        super(R1Regularization, self).__init__()

    def forward(
        self,
        d_out: torch.Tensor,
        real_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            d_out: critic output on real samples, shape [B]
            real_data: real trajectories, requires_grad=True, shape [B, T, D]

        Returns:
            scalar R1 penalty
        """
        grad_real = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(dim=1)
        return grad_penalty.mean()
