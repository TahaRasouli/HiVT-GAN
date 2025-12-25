import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from losses import AdversarialDiscriminatorLoss
from losses import AdversarialGeneratorLoss

from metrics import ADE
from metrics import FDE
from metrics import MR

# Optional realism metrics (only if you added them in metrics/__init__.py)
try:
    from metrics import Jerk, SpeedViolation, EndpointDiversity
    _HAS_REALISM_METRICS = True
except Exception:
    _HAS_REALISM_METRICS = False

from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder

from models.critics import ShortScaleCritic, MidScaleCritic, LongScaleCritic

from utils import TemporalData


class HiVT(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,

                 # GAN args (passed from train.py)
                 use_gan: bool = False,
                 lambda_adv: float = 0.1,
                 lambda_r1: float = 1.0,
                 critic_steps: int = 1,
                 short_horizon: int = 10,
                 mid_horizon: int = 20,
                 critic_lr: float = 1e-4,
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters()

        # Core settings
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel

        # Optim settings
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        # GAN settings
        self.use_gan = use_gan
        self.lambda_adv = lambda_adv
        self.lambda_r1 = lambda_r1
        self.critic_steps = critic_steps
        self.short_horizon = short_horizon
        self.mid_horizon = mid_horizon
        self.critic_lr = critic_lr

        # If GAN is enabled, we need manual optimization for two optimizers
        self.automatic_optimization = not self.use_gan

        # -------------------------
        # Generator (HiVT) modules
        # -------------------------
        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)

        # Supervised losses
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        # Metrics (original)
        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()

        # Optional realism metrics
        if _HAS_REALISM_METRICS:
            self.val_jerk = Jerk()
            self.val_speed_violation = SpeedViolation()
            self.val_endpoint_diversity = EndpointDiversity()


        # -------------------------
        # Critics + adversarial losses
        # -------------------------
        if self.use_gan:
            # Multi-scale critics (SN is applied inside their layers)
            self.D_short = ShortScaleCritic(horizon=self.short_horizon, input_dim=2, hidden_dim=128)
            self.D_mid = MidScaleCritic(horizon=self.mid_horizon, input_dim=2, hidden_dim=256)
            self.D_long = LongScaleCritic(horizon=self.future_steps, input_dim=2, hidden_dim=256)

            self.critics = {
                "short": self.D_short,
                "mid": self.D_mid,
                "long": self.D_long,
            }

            self.d_loss_fn = AdversarialDiscriminatorLoss(lambda_r1=self.lambda_r1)
            self.g_loss_fn = AdversarialGeneratorLoss(lambda_adv=self.lambda_adv)

    # ---------------------------------------------------------
    # Forward pass (generator only)
    # ---------------------------------------------------------
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
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return y_hat, pi

    # ---------------------------------------------------------
    # Helper: supervised loss (unchanged)
    # ---------------------------------------------------------
    def _supervised_losses(self, data: TemporalData, y_hat: torch.Tensor, pi: torch.Tensor):
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]  # [N, H]
        valid_steps = reg_mask.sum(dim=-1)                           # [N]
        cls_mask = valid_steps > 0

        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]

        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])

        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)

        return reg_loss, cls_loss, best_mode

    # ---------------------------------------------------------
    # Helper: build critic inputs
    # ---------------------------------------------------------
    def _build_real_fake_dicts(self, data: TemporalData, y_hat: torch.Tensor, best_mode: torch.Tensor):
        """
        Critics operate on absolute future positions.
        Here we compare:
            real: data.y [N, H, 2]
            fake: best-mode predicted future (positions) [N, H, 2]

        Note: y_hat stores [F, N, H, 4] when uncertain; we take [:2].
        """
        real = data.y  # [N, H, 2]
        fake = y_hat[best_mode, torch.arange(data.num_nodes), :, :2]  # [N, H, 2]

        # Filter out actors with no valid future at all
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]  # [N, H]
        keep = reg_mask.any(dim=1)  # [N]
        real = real[keep]
        fake = fake[keep]

        # If empty batch (edge case), return empty dicts
        if real.numel() == 0:
            empty = torch.zeros((0, self.future_steps, 2), device=self.device)
            return (
                {"short": empty[:, :self.short_horizon],
                 "mid": empty[:, :self.mid_horizon],
                 "long": empty},
                {"short": empty[:, :self.short_horizon],
                 "mid": empty[:, :self.mid_horizon],
                 "long": empty},
                keep,
            )

        real_trajs = {
            "short": real[:, :self.short_horizon],
            "mid": real[:, :self.mid_horizon],
            "long": real,
        }
        fake_trajs = {
            "short": fake[:, :self.short_horizon],
            "mid": fake[:, :self.mid_horizon],
            "long": fake,
        }
        return real_trajs, fake_trajs, keep

    # ---------------------------------------------------------
    # Training step
    # ---------------------------------------------------------
    def training_step(self, data, batch_idx):
        # -------------------------
        # Standard supervised training (no GAN)
        # -------------------------
        if not self.use_gan:
            y_hat, pi = self(data)
            reg_loss, cls_loss, _ = self._supervised_losses(data, y_hat, pi)
            loss = reg_loss + cls_loss
            self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            return loss

        # -------------------------
        # GAN training (manual optimization)
        # -------------------------
        opt_g, opt_d = self.optimizers()

        # 1) Forward generator
        y_hat, pi = self(data)
        reg_loss, cls_loss, best_mode = self._supervised_losses(data, y_hat, pi)

        # Build critic inputs
        real_trajs, fake_trajs, keep = self._build_real_fake_dicts(data, y_hat, best_mode)

        # If there are no valid actors with future, skip adversarial and train supervised only
        if real_trajs["long"].size(0) == 0:
            g_total = reg_loss + cls_loss
            opt_g.zero_grad()
            self.manual_backward(g_total)
            opt_g.step()

            self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_g_total', g_total, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            return

        # 2) Update critics multiple times
        for _ in range(self.critic_steps):
            d_loss, d_logs = self.d_loss_fn(self.critics, real_trajs, fake_trajs)

            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

        # 3) Generator adversarial loss
        g_adv, g_logs = self.g_loss_fn(self.critics, fake_trajs)

        # 4) Total generator loss
        g_total = reg_loss + cls_loss + g_adv

        opt_g.zero_grad()
        self.manual_backward(g_total)
        opt_g.step()

        # Logging
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_g_adv', g_adv, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_g_total', g_total, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_d_loss', d_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        for k, v in d_logs.items():
            self.log(f"train_{k}", v, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        for k, v in g_logs.items():
            self.log(f"train_{k}", v, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

    # ---------------------------------------------------------
    # Validation (unchanged + optional realism metrics)
    # ---------------------------------------------------------
    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)

        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        # Standard agent metrics
        y_hat_agent = y_hat[:, data['agent_index'], :, :2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]

        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

        # Optional realism metrics on the best agent trajectory
        if _HAS_REALISM_METRICS:
            self.val_jerk.update(y_hat_best_agent)
            self.val_speed_violation.update(y_hat_best_agent)
            self.val_endpoint_diversity.update(y_hat_agent)  # uses all modes for diversity

            self.log('val_jerk', self.val_jerk, prog_bar=False, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
            self.log('val_speed_violation', self.val_speed_violation, prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=y_agent.size(0))
            self.log('val_endpoint_diversity', self.val_endpoint_diversity, prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=y_agent.size(0))

    # ---------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------
    def configure_optimizers(self):
        # --- Generator optimizer (original grouping logic) ---
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)

        param_dict = {param_name: param for param_name, param in self.named_parameters()}

        # If GAN enabled, exclude critics from generator param dict grouping
        if self.use_gan:
            critic_param_names = set()
            for n, _ in self.D_short.named_parameters():
                critic_param_names.add(f"D_short.{n}")
            for n, _ in self.D_mid.named_parameters():
                critic_param_names.add(f"D_mid.{n}")
            for n, _ in self.D_long.named_parameters():
                critic_param_names.add(f"D_long.{n}")

            # Remove critic params from decay/no_decay
            decay = {n for n in decay if n not in critic_param_names}
            no_decay = {n for n in no_decay if n not in critic_param_names}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        opt_g = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.T_max, eta_min=0.0)

        if not self.use_gan:
            return [opt_g], [sch_g]

        # --- Critic optimizer (separate) ---
        d_params = list(self.D_short.parameters()) + list(self.D_mid.parameters()) + list(self.D_long.parameters())
        opt_d = torch.optim.AdamW(d_params, lr=self.critic_lr, weight_decay=0.0)

        # No scheduler for critics by default (keeps things stable)
        return [opt_g, opt_d], [sch_g]

    # ---------------------------------------------------------
    # Argparse
    # ---------------------------------------------------------
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)

        # GAN args (optional)
        parent_parser.add_argument('--use_gan', action='store_true')
        parent_parser.add_argument('--lambda_adv', type=float, default=0.1)
        parent_parser.add_argument('--lambda_r1', type=float, default=1.0)
        parent_parser.add_argument('--critic_steps', type=int, default=1)
        parent_parser.add_argument('--critic_lr', type=float, default=1e-4)
        parent_parser.add_argument('--short_horizon', type=int, default=10)
        parent_parser.add_argument('--mid_horizon', type=int, default=20)

        return parent_parser
