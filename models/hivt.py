import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss, SoftTargetCrossEntropyLoss
from losses import AdversarialDiscriminatorLoss, AdversarialGeneratorLoss
from metrics import ADE, FDE, MR

try:
    from metrics import Jerk, SpeedViolation, EndpointDiversity
    _HAS_REALISM_METRICS = True
except Exception:
    _HAS_REALISM_METRICS = False

from models import GlobalInteractor, LocalEncoder, MLPDecoder
from models.critics import ShortScaleCritic, MidScaleCritic, LongScaleCritic
from utils import TemporalData

class HiVT(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters()
        
        # Core & Optim Params
        self.historical_steps = kwargs.get("historical_steps", 20)
        self.future_steps = kwargs.get("future_steps", 30)
        self.num_modes = kwargs.get("num_modes", 6)
        self.rotate = kwargs.get("rotate", True)
        self.lr = kwargs.get("lr", 5e-4)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.T_max = kwargs.get("T_max", 64)

        # GAN settings
        self.use_gan = kwargs.get("use_gan", False)
        self.automatic_optimization = not self.use_gan
        self.lambda_adv = kwargs.get("lambda_adv", 0.1)
        self.lambda_r1 = kwargs.get("lambda_r1", 1.0)
        self.critic_steps = kwargs.get("critic_steps", 1)
        self.critic_lr = kwargs.get("critic_lr", 1e-4)

        # Modules
        self.local_encoder = LocalEncoder(historical_steps=self.historical_steps, **kwargs)
        self.global_interactor = GlobalInteractor(historical_steps=self.historical_steps, **kwargs)
        self.decoder = MLPDecoder(local_channels=kwargs['embed_dim'], global_channels=kwargs['embed_dim'], future_steps=self.future_steps, num_modes=self.num_modes)

        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        # Metrics
        self.minADE, self.minFDE, self.minMR = ADE(), FDE(), MR()
        if _HAS_REALISM_METRICS:
            self.val_jerk = Jerk()
            self.val_speed_violation = SpeedViolation()
            self.val_endpoint_diversity = EndpointDiversity()

        if self.use_gan:
            self.D_short = ShortScaleCritic(horizon=kwargs.get("short_horizon", 10))
            self.D_mid = MidScaleCritic(horizon=kwargs.get("mid_horizon", 20))
            self.D_long = LongScaleCritic(horizon=self.future_steps)
            self.critics = {"short": self.D_short, "mid": self.D_mid, "long": self.D_long}
            self.d_loss_fn = AdversarialDiscriminatorLoss(lambda_r1=self.lambda_r1)
            self.g_loss_fn = AdversarialGeneratorLoss(lambda_adv=1.0) # We scale by lambda_adv in training_step

    def forward(self, data):
        # (Keep your existing rotation logic here)
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        return self.decoder(local_embed=local_embed, global_embed=global_embed)

    def training_step(self, data, batch_idx):
        if torch.isnan(data.y).any(): return self.local_encoder.parameters().__next__().sum() * 0.0
        
        if not self.use_gan:
            y_hat, pi = self(data)
            reg_loss, cls_loss, _ = self._supervised_losses(data, y_hat, pi)
            loss = reg_loss + cls_loss
            self.log("train_reg_loss", reg_loss, on_epoch=True, prog_bar=True, batch_size=data.num_graphs)
            return loss

        # --- GAN CASE (Manual Optimization) ---
        opt_g, opt_d = self.optimizers()
        y_hat, pi = self(data)

        reg_loss, cls_loss, best_mode = self._supervised_losses(data, y_hat, pi)
        # WTA: Critics only see the best mode to preserve minFDE accuracy
        real_trajs, fake_trajs, keep = self._build_real_fake_dicts(data, y_hat, best_mode)
        has_valid = (real_trajs["long"].size(0) > 0)

        # Discriminator Step
        for _ in range(self.critic_steps):
            d_loss, d_logs = self.d_loss_fn(self.critics, real_trajs, fake_trajs) if has_valid else (y_hat.sum()*0, {})
            opt_d.zero_grad(); self.manual_backward(d_loss); opt_d.step()

        # Generator Step
        g_adv, g_logs = self.g_loss_fn(self.critics, fake_trajs) if has_valid else (y_hat.sum()*0, {})
        g_total = reg_loss + cls_loss + (self.lambda_adv * g_adv)
        opt_g.zero_grad(); self.manual_backward(g_total); opt_g.step()

        # Weighted Diversity Log
        if _HAS_REALISM_METRICS and has_valid:
            self.val_endpoint_diversity.update(y_hat[:, data['agent_index'], :, :2], pi[data['agent_index']])

        self.log("train_reg_loss", reg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return g_total

    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        # (Calculate metrics as before)
        # ...
        if _HAS_REALISM_METRICS:
            self.val_endpoint_diversity.update(y_hat[:, data['agent_index'], :, :2], pi[data['agent_index']])
        # (Log metrics)

    def configure_optimizers(self):
        # (Use the TTUR logic with betas=(0.5, 0.9) we discussed)
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        # (Keep your existing ArgParser logic)
        return parent_parser