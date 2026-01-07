import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import ADE, FDE, MR
from models import GlobalInteractor, LocalEncoder
from models import DiffusionDecoder
from utils import VarianceSchedule

class DiVT(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Reuse HiVT Encoders
        self.historical_steps = kwargs.get("historical_steps", 20)
        embed_dim = kwargs.get("embed_dim", 128)
        
        self.local_encoder = LocalEncoder(
            historical_steps=self.historical_steps,
            node_dim=2, edge_dim=2, embed_dim=embed_dim,
            num_heads=8, dropout=0.1, num_temporal_layers=4,
            local_radius=50
        )
        self.global_interactor = GlobalInteractor(
            historical_steps=self.historical_steps,
            embed_dim=embed_dim, edge_dim=2, num_modes=1,
            num_heads=8, num_layers=3, dropout=0.1
        )
        
        # 2. Diffusion Specifics
        self.diff_steps = kwargs.get("diff_steps", 100)
        self.scheduler = VarianceSchedule(num_steps=self.diff_steps)
        self.decoder = DiffusionDecoder(embed_dim=embed_dim, future_steps=30)
        
        # 3. Metrics
        self.minADE = ADE()
        self.minFDE = FDE()
        self.val_minADE = ADE() # Separate instance for validation logging
        self.val_minFDE = FDE()

    def forward(self, data):
        if self.hparams.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            
            # Rotate ground truth if available (crucial for loss calc later)
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        # Now it is safe to call LocalEncoder
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        return global_embed

    def training_step(self, data, batch_idx):
        # A. Get Context (The Condition)
        context = self(data) # [Batch, Embed_Dim]

        # If we have [Modes, Batch, Nodes, Embed], we want [Nodes, Embed]
        if context.dim() == 4:
            # Squeeze dim 0 (Modes) and dim 1 (Batch)
            context = context.view(-1, self.hparams.embed_dim)
        
        # B. Get Ground Truth Future
        y_gt = data.y # [Batch, 30, 2]
        
        # C. Sample Random Noise & Timestep
        noise = torch.randn_like(y_gt)
        t = torch.randint(0, self.diff_steps, (y_gt.size(0),), device=self.device)
        
        # D. Add Noise (Forward Diffusion)
        x_noisy = self.scheduler.add_noise(y_gt, noise, t)
        
        # E. Predict Noise (Reverse Diffusion)
        # The model tries to guess what noise was added
        noise_pred = self.decoder(x_noisy, t, context)
        
        # F. Loss: Simple MSE
        # Mask out padding to avoid learning from zeros
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        loss = F.mse_loss(noise_pred[reg_mask], noise[reg_mask])
        
        self.log("train_diff_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        context = self(data)
        context = context.reshape(-1, self.hparams.embed_dim)
        B = context.size(0)
        
        # SAMPLING LOOP
        x = torch.randn(B, 30, 2, device=self.device)
        
        for t in reversed(range(self.diff_steps)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.decoder(x, t_batch, context)
            
            alpha = self.scheduler.alphas[t]
            alpha_cumprod = self.scheduler.alphas_cumprod[t]
            beta = self.scheduler.betas[t]
            
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
            mean = coef1 * (x - coef2 * noise_pred)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

        # --- METRICS FIX ---
        valid_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_mask_fde = valid_mask[:, -1]
        
        # Filter predictions and targets
        x_filtered = x[valid_mask_fde]       # Shape: [N_valid, 30, 2]
        y_filtered = data.y[valid_mask_fde]  # Shape: [N_valid, 30, 2]
        
        # REMOVED: x_filtered = x_filtered.unsqueeze(0) 
        # We pass 3D tensors directly because your metric expects [Batch, Time, 2]
        
        if x_filtered.size(0) > 0: 
            self.val_minADE.update(x_filtered, y_filtered)
            self.val_minFDE.update(x_filtered, y_filtered)
        
        self.log("val_minADE", self.val_minADE, prog_bar=True, batch_size=B)
        self.log("val_minFDE", self.val_minFDE, prog_bar=True, batch_size=B)

    def on_validation_epoch_end(self):
        """
        Prints a clean summary of the epoch's performance to the terminal,
        keeping a permanent log visible to the user.
        """
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            # We fetch the metrics from the trainer's dictionary
            ade = metrics.get('val_minADE', 0.0)
            fde = metrics.get('val_minFDE', 0.0)
            loss = metrics.get('train_diff_loss', 0.0)
            
            print(f"\nEpoch {self.current_epoch:03d} | "
                  f"Train Loss: {loss:.4f} | "
                  f"val_minADE: {ade:.4f} | "
                  f"val_minFDE: {fde:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)
        return [optimizer], [scheduler]
