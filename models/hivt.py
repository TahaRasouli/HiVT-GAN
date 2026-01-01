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

            reg_mask = ~data['padding_mask'][:, self.historical_steps:]
            if reg_mask.sum() == 0:
                return 

            # 1. Best-mode selection for regression loss
            l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
            best_mode = l2_norm.argmin(dim=0)
            y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
            
            reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
            self.log('val_reg_loss', reg_loss, prog_bar=True, on_epoch=True, sync_dist=True)

            # 2. Agent-specific metrics (minADE/minFDE)
            y_hat_agent = y_hat[:, data['agent_index'], :, :2]
            y_agent = data.y[data['agent_index']]
            
            fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
            best_mode_agent = fde_agent.argmin(dim=0)
            y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]

            self.minADE.update(y_hat_best_agent, y_agent)
            self.minFDE.update(y_hat_best_agent, y_agent)
            self.minMR.update(y_hat_best_agent, y_agent)
            
            self.log('val_minADE', self.minADE, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('val_minFDE', self.minFDE, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('val_minMR', self.minMR, prog_bar=True, on_epoch=True, sync_dist=True)

            # 3. Realism Metrics
            if _HAS_REALISM_METRICS:
                self.val_jerk.update(y_hat_best_agent)
                self.val_speed_violation.update(y_hat_best_agent)
                # Weighted diversity using mode probabilities
                pi_agent = pi[data['agent_index']]
                self.val_endpoint_diversity.update(y_hat_agent, pi_agent)

                self.log('val_jerk', self.val_jerk, on_epoch=True, sync_dist=True)
                self.log('val_speed_violation', self.val_speed_violation, on_epoch=True, sync_dist=True)
                self.log('val_endpoint_diversity', self.val_endpoint_diversity, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
            if not self.use_gan:
                opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.T_max, eta_min=1e-6)
                return {
                    "optimizer": opt,
                    "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
                }

            # GAN Optimizers
            gen_params = (list(self.local_encoder.parameters()) + 
                        list(self.global_interactor.parameters()) + 
                        list(self.decoder.parameters()))
            
            disc_params = (list(self.D_short.parameters()) + 
                        list(self.D_mid.parameters()) + 
                        list(self.D_long.parameters()))

            # Generator Optimizer (betas 0.5, 0.9 for stability)
            opt_g = torch.optim.AdamW(gen_params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.9))
            # Discriminator Optimizer (Higher learning rate often helps)
            opt_d = torch.optim.AdamW(disc_params, lr=self.critic_lr, weight_decay=1e-4, betas=(0.5, 0.9))

            sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.T_max, eta_min=1e-6)

            return [opt_g, opt_d], [{"scheduler": sch_g, "interval": "epoch"}]

    @staticmethod
        def add_model_specific_args(parent_parser):
            parser = parent_parser.add_argument_group('HiVT')
            parser.add_argument('--historical_steps', type=int, default=20)
            parser.add_argument('--future_steps', type=int, default=30)
            parser.add_argument('--num_modes', type=int, default=6)
            parser.add_argument('--rotate', type=bool, default=True)
            parser.add_argument('--node_dim', type=int, default=2)
            parser.add_argument('--edge_dim', type=int, default=2)
            parser.add_argument('--embed_dim', type=int, default=128)
            parser.add_argument('--num_heads', type=int, default=8)
            parser.add_argument('--dropout', type=float, default=0.1)
            parser.add_argument('--num_temporal_layers', type=int, default=4)
            parser.add_argument('--num_global_layers', type=int, default=3)
            parser.add_argument('--local_radius', type=float, default=50)
            parser.add_argument('--parallel', type=bool, default=False)
            parser.add_argument('--lr', type=float, default=5e-4)
            parser.add_argument('--weight_decay', type=float, default=1e-4)
            parser.add_argument('--T_max', type=int, default=64)

            # GAN Arguments
            parser.add_argument('--use_gan', action='store_true')
            parser.add_argument('--lambda_adv', type=float, default=0.1)
            parser.add_argument('--lambda_r1', type=float, default=1.0)
            parser.add_argument('--critic_steps', type=int, default=1)
            parser.add_argument('--critic_lr', type=float, default=1e-4)
            parser.add_argument('--short_horizon', type=int, default=10)
            parser.add_argument('--mid_horizon', type=int, default=20)
            
            return parent_parser