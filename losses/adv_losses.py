import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.r1_regularization import R1Regularization


class AdversarialDiscriminatorLoss(nn.Module):
    """
    Multi-scale hinge GAN discriminator loss with R1 regularization.
    """

    def __init__(
        self,
        lambda_r1: float = 1.0,
    ):
        super(AdversarialDiscriminatorLoss, self).__init__()
        self.lambda_r1 = lambda_r1
        self.r1_reg = R1Regularization()

    def forward(
        self,
        critics: dict,
        real_trajs: dict,
        fake_trajs: dict,
    ):
        """
        Args:
            critics: dict {name: critic_module}
            real_trajs: dict {name: real_traj_tensor}
            fake_trajs: dict {name: fake_traj_tensor}

        Returns:
            total_loss: scalar
            log_dict: dict of per-scale losses
        """
        total_loss = 0.0
        log_dict = {}

        for name, critic in critics.items():
            real = real_trajs[name].requires_grad_(True)
            fake = fake_trajs[name].detach()

            d_real = critic(real)
            d_fake = critic(fake)

            loss_real = F.relu(1.0 - d_real).mean()
            loss_fake = F.relu(1.0 + d_fake).mean()
            d_loss = loss_real + loss_fake

            r1 = self.r1_reg(d_real, real)
            d_loss = d_loss + self.lambda_r1 * r1

            total_loss = total_loss + d_loss

            log_dict[f"d_loss_{name}"] = d_loss.detach()
            log_dict[f"r1_{name}"] = r1.detach()

        return total_loss, log_dict



class AdversarialGeneratorLoss(nn.Module):
    def __init__(self, lambda_adv: float = 1.0, lambda_feat: float = 0.5):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat

    def forward(self, critics: dict, fake_trajs: dict):
        total_loss = 0.0
        log_dict = {}

        for name, critic in critics.items():
            fake = fake_trajs[name]
            # Standard Adversarial Loss
            g_loss = -critic(fake).mean()
            
            # Optional: Feature Matching (if the critic returns intermediate features)
            # This is the "Secret Sauce" to beat baselines.
            total_loss = total_loss + g_loss
            log_dict[f"g_loss_{name}"] = g_loss.detach()

        return self.lambda_adv * total_loss, log_dictloss
        return total_loss, log_dict
