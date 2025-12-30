import torch
from torchmetrics import Metric

class EndpointDiversity(Metric):
    """
    Weighted pairwise distance between final endpoints across modes.
    Incorporates mode probabilities (pi) to penalize diversity in low-confidence modes.
    """

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, trajs: torch.Tensor, pi: torch.Tensor):
        """
        trajs: [F, B, T, 2]  (multi-modal predictions)
        pi:    [B, F]        (probabilities per mode, log_probs or softmax)
        """
        F = trajs.size(0)
        if F < 2:
            return

        # Convert pi to probabilities if they are log-probs
        if (pi < 0).any():
            probs = torch.exp(pi)
        else:
            probs = pi

        endpoints = trajs[:, :, -1]  # [F, B, 2]
        B = endpoints.size(1)

        total_dist = 0.0
        weight_sum = 0.0

        # Calculate probability-weighted pairwise distance
        for i in range(F):
            for j in range(i + 1, F):
                # Distance between mode i and mode j
                d_ij = torch.norm(endpoints[i] - endpoints[j], dim=-1) # [B]
                
                # Weight by the product of their probabilities
                # If both modes have high probability, their diversity is "meaningful"
                w_ij = probs[:, i] * probs[:, j] # [B]
                
                total_dist += (d_ij * w_ij).sum()
                weight_sum += w_ij.sum()

        # Avoid division by zero if all probabilities collapsed
        if weight_sum > 1e-6:
            self.sum += total_dist / weight_sum
            self.count += 1

    def compute(self):
        return self.sum / self.count