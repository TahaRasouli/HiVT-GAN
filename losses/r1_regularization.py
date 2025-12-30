import torch
import torch.nn as nn


class R1Regularization(nn.Module):
    """
    R1 regularization normalized by input dimensionality.
    """
    def __init__(self):
        super(R1Regularization, self).__init__()

    def forward(self, d_out: torch.Tensor, real_data: torch.Tensor) -> torch.Tensor:
        grad_real = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=False,
            only_inputs=True,
        )[0]

        # Calculate the number of elements (T * D) to normalize the penalty
        num_elements = grad_real.shape[1] * grad_real.shape[2]
        
        # Calculate squared norm of the gradient
        grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(dim=1)
        
        # Return the mean penalty across the batch, normalized by horizon
        return grad_penalty.mean() / num_elements