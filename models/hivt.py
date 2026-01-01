import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

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
        
        # 1. Extract HiVT Core Parameters
        self.historical_steps = kwargs.get("historical_steps", 20)
        self.future_steps = kwargs.get("future_steps", 30)
        self.num_modes = kwargs.get("num_modes", 6)
        self.rotate = kwargs.get("rotate", True)
        self.lr = kwargs.get("lr", 5e-4)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.T_max = kwargs.get("T_max", 64)
        
        embed_dim = kwargs.get("embed_dim", 128)
        node_dim = kwargs.get("node_dim", 2)
        edge_dim = kwargs.get("edge_dim", 2)
        dropout = kwargs.get("dropout", 0.1)
        num_heads = kwargs.get("num_heads", 8)

        # 2. GAN settings
        self.use_gan = kwargs.get("use_gan", False)
        self.lambda_adv = kwargs.get("lambda_adv", 0.1)
        self.lambda_r1 = kwargs.get("lambda_r1", 1.0)
        self.critic_steps = kwargs.get("critic_steps", 1)
        self.critic_lr = kwargs.get("critic_lr", 1e-4)
        self.automatic_optimization = not self.use_gan

        # 3. Initialize Modules with STRICT arguments
        # We no longer pass **kwargs to avoid "unexpected keyword argument" errors
        self.local_encoder = LocalEncoder(
            historical_steps=self.historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_temporal_layers=kwargs.get("num_temporal_layers", 4),
            local_radius=kwargs.get("local_radius", 50),
            parallel=kwargs.get("parallel", False)
        )
        
        self.global_interactor = GlobalInteractor(
            historical_steps=self.historical_steps,
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            num_modes=self.num_modes,
            num_heads=num_heads,
            num_layers=kwargs.get("num_global_layers", 3),
            dropout=dropout,
            rotate=self.rotate
        )
        
        self.decoder = MLPDecoder(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=self.future_steps,
            num_modes=self.num_modes
        )

        # (Rest of your supervised losses and metrics initialization stays the same)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        # 4. Metrics
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
            self.g_loss_fn = AdversarialGeneratorLoss(lambda_adv=1.0)

    def forward(self, data: TemporalData):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        return self.decoder(local_embed=local_embed, global_embed=global_embed)

    def _supervised_losses(self, data, y_hat, pi):
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        
        # Soft target for classification based on distance
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        avg_l2 = -l2_norm[:, cls_mask] / (valid_steps[cls_mask] + 1e-6)
        soft_target = F.softmax(avg_l2, dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        return reg_loss, cls_loss, best_mode

    def _build_real_fake_dicts(self, data, y_hat, best_mode):
        real = data.y
        fake = y_hat[best_mode, torch.arange(data.num_nodes), :, :2]
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        keep = reg_mask.any(dim=1)
        
        real_trajs = {"short": real[keep, :10], "mid": real[keep, :20], "long": real[keep]}
        fake_trajs = {"short": fake[keep, :10], "mid": fake[keep, :20], "long": fake[keep]}
        return real_trajs, fake_trajs, keep

    def training_step(self, data, batch_idx):
        # 1. Setup flags
        # We calculate these but DO NOT branch yet.
        has_nans = torch.isnan(data.y).any()
        
        # -----------------------------------------------------
        # Case A: Supervised Only
        # -----------------------------------------------------
        if not self.use_gan:
            if has_nans:
                # Dummy loss to keep DDP in sync
                dummy_loss = sum(p.sum() for p in self.parameters()) * 0.0
                # Log 0.0 to keep sync_dist happy
                self.log("train_reg_loss", 0.0, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
                return dummy_loss
            
            y_hat, pi = self(data)
            reg_loss, cls_loss, _ = self._supervised_losses(data, y_hat, pi)
            loss = reg_loss + cls_loss
            self.log("train_reg_loss", reg_loss, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
            return loss

        # -----------------------------------------------------
        # Case B: GAN training (Manual Optimization)
        # -----------------------------------------------------
        opt_g, opt_d = self.optimizers()
        
        # Initialize variables to ensure they exist for logging later
        d_loss_val = torch.tensor(0.0, device=self.device)
        reg_loss_val = torch.tensor(0.0, device=self.device)
        
        # -----------------------------------------------------
        # 1. Forward Pass & Validity Check
        # -----------------------------------------------------
        # We wrap forward pass to prevent crashing on NaNs
        if not has_nans:
            y_hat, pi = self(data)
            reg_loss, cls_loss, best_mode = self._supervised_losses(data, y_hat, pi)
            real_trajs, fake_trajs, keep = self._build_real_fake_dicts(data, y_hat, best_mode)
            is_valid = (real_trajs["long"].size(0) > 0)
            
            if is_valid:
                reg_loss_val = reg_loss.detach()
        else:
            is_valid = False
            y_hat = None

        # -----------------------------------------------------
        # 2. D Step (Discriminator)
        # -----------------------------------------------------
        for _ in range(self.critic_steps):
            opt_d.zero_grad()
            
            if is_valid:
                # Valid Path: Detach and Calculate
                fake_detached = {k: v.detach().requires_grad_(True) for k, v in fake_trajs.items()}
                d_loss, _ = self.d_loss_fn(self.critics, real_trajs, fake_detached)
                self.manual_backward(d_loss)
                self.clip_gradients(opt_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                d_loss_val = d_loss.detach()
            else:
                # Dummy Path: Touch parameters to satisfy DDP
                # We sum *all* critic params to ensure DDP buckets are happy
                dummy = sum(p.sum() for p in self.D_short.parameters()) 
                dummy += sum(p.sum() for p in self.D_mid.parameters())
                dummy += sum(p.sum() for p in self.D_long.parameters())
                self.manual_backward(dummy * 0.0)
            
            # ALWAYS STEP
            opt_d.step()

        # -----------------------------------------------------
        # 3. G Step (Generator)
        # -----------------------------------------------------
        opt_g.zero_grad()
        
        if is_valid:
            # Valid Path
            g_adv, _ = self.g_loss_fn(self.critics, fake_trajs)
            g_total = reg_loss + cls_loss + (self.lambda_adv * g_adv)
            self.manual_backward(g_total)
            self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        else:
            # Dummy Path
            dummy = sum(p.sum() for p in self.local_encoder.parameters()) * 0.0
            self.manual_backward(dummy)
            g_total = dummy
            
        # ALWAYS STEP
        opt_g.step()

        # -----------------------------------------------------
        # 4. Scheduler Step
        # -----------------------------------------------------
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

        # -----------------------------------------------------
        # 5. Logging (UNCONDITIONAL)
        # -----------------------------------------------------
        # The logging call happens NO MATTER WHAT.
        # This prevents the synchronization deadlock.
        self.log("train_reg_loss", reg_loss_val, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=data.num_graphs)
        self.log("train_d_loss", d_loss_val, on_epoch=True, sync_dist=True, batch_size=data.num_graphs)
        
        # Realism metrics are local (no sync needed), so we can keep them conditional
        if _HAS_REALISM_METRICS and is_valid:
             self.val_endpoint_diversity.update(
                y_hat[:, data['agent_index'], :, :2].detach(), 
                pi[data['agent_index']].detach()
            )

        return g_total
    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)

        # 1. Standard Regression Logic
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        
        # Log Loss
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        # 2. Slice for Agent (The "Interesting" Vehicle)
        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        
        # Calculate ADE/FDE
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]

        # Update Standard Metrics
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

        # 3. REALISM METRICS (Enable this NOW)
        if _HAS_REALISM_METRICS:
            # Jerk & Speed: Check if the "best" trajectory is physically realistic
            self.val_jerk.update(y_hat_best_agent, y_agent)
            self.val_speed_violation.update(y_hat_best_agent, y_agent)
            
            # Diversity: Check if the 6 modes are distinct (Crucial for GANs)
            self.val_endpoint_diversity.update(y_hat_agent, pi[data['agent_index']])

            self.log('val_jerk', self.val_jerk, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
            self.log('val_speed', self.val_speed_violation, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
            self.log('val_div', self.val_endpoint_diversity, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            print(f"\nEpoch {self.current_epoch:03d} | val_minADE: {metrics.get('val_minADE'):.4f} | val_minFDE: {metrics.get('val_minFDE'):.4f}")

    def configure_optimizers(self):
        if not self.use_gan:
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.T_max, eta_min=1e-6)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

        opt_g = torch.optim.AdamW(
            list(self.local_encoder.parameters()) + list(self.global_interactor.parameters()) + list(self.decoder.parameters()),
            lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.9)
        )
        opt_d = torch.optim.AdamW(
            list(self.D_short.parameters()) + list(self.D_mid.parameters()) + list(self.D_long.parameters()),
            lr=self.critic_lr, weight_decay=1e-4, betas=(0.5, 0.9)
        )
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.T_max, eta_min=1e-6)
        return [opt_g, opt_d], [{"scheduler": sch_g, "interval": "epoch"}]

    def _build_real_fake_dicts(self, data, y_hat, best_mode):
        # Ensure we only take the (x, y) coordinates for the critics
        real = data.y
        fake = y_hat[best_mode, torch.arange(data.num_nodes), :, :2]
        
        # Do NOT use .item() or .numpy() here, stay in torch.Tensor
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        keep = reg_mask.any(dim=1)
        
        real_trajs = {
            "short": real[keep, :10],
            "mid": real[keep, :20],
            "long": real[keep]
        }
        fake_trajs = {
            "short": fake[keep, :10],
            "mid": fake[keep, :20],
            "long": fake[keep]
        }
        return real_trajs, fake_trajs, keep

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
        parser.add_argument('--use_gan', action='store_true')
        parser.add_argument('--lambda_adv', type=float, default=0.1)
        parser.add_argument('--lambda_r1', type=float, default=1.0)
        parser.add_argument('--critic_steps', type=int, default=1)
        parser.add_argument('--critic_lr', type=float, default=1e-4)
        parser.add_argument('--short_horizon', type=int, default=10)
        parser.add_argument('--mid_horizon', type=int, default=20)
        return parent_parser