import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.r1_regularization import R1Regularization


class AdversarialDiscriminatorLoss(nn.Module):
    def __init__(self, lambda_r1: float = 1.0):
        super(AdversarialDiscriminatorLoss, self).__init__()
        self.lambda_r1 = lambda_r1
        self.r1_reg = R1Regularization()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, critics: dict, real_trajs: dict, fake_trajs: dict):
        total_loss = 0.0
        logs = {}

        for scale in critics.keys():
            # 1. Real Data Forward Pass
            # We must enable gradients on input for R1 to work
            real_in = real_trajs[scale].detach().requires_grad_(True)
            d_real = critics[scale](real_in)
            
            # 2. Fake Data Forward Pass
            d_fake = critics[scale](fake_trajs[scale])

            # 3. Standard GAN Loss (Hinge or BCE)
            # Using BCE here for stability with your setup
            real_loss = self.bce(d_real, torch.ones_like(d_real))
            fake_loss = self.bce(d_fake, torch.zeros_like(d_fake))
            
            # 4. R1 Regularization
            # (Only applied to real data)
            r1_loss = self.r1_reg(d_real, real_in)
            
            # Combine
            scale_loss = real_loss + fake_loss + (self.lambda_r1 * r1_loss)
            total_loss += scale_loss
            
            logs[f"d_loss_{scale}"] = scale_loss.detach()
            logs[f"d_r1_{scale}"] = r1_loss.detach()

        return total_loss, logs

class AdversarialGeneratorLoss(nn.Module):
    def __init__(self, lambda_adv: float = 1.0):
        super(AdversarialGeneratorLoss, self).__init__()
        self.lambda_adv = lambda_adv # Note: usually handled in training_step, but kept here for consistency
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, critics: dict, fake_trajs: dict):
        total_loss = 0.0
        logs = {}

        for scale in critics.keys():
            d_fake = critics[scale](fake_trajs[scale])
            # Generator wants Discriminator to output 1 (Real)
            scale_loss = self.bce(d_fake, torch.ones_like(d_fake))
            total_loss += scale_loss
            logs[f"g_loss_{scale}"] = scale_loss.detach()

        return total_loss, logs