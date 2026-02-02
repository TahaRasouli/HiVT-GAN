import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

# ------------------------------------------------------------------------------
# 1. FOCAL LOSS
#    Forces the model to focus on "Hard" examples (Lane Changes/U-Turns) 
#    rather than being satisfied with getting the easy ones right.
# ------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # FIX: Register alpha as a buffer so it moves to GPU automatically
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # Calculate standard Cross Entropy first
        # reduction='none' so we can apply the focal weight per-sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calculate probability of the correct class (pt)
        pt = torch.exp(-ce_loss)
        
        # Apply Focal Term: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ------------------------------------------------------------------------------
# 2. TEMPORAL HEADING EXTRACTOR (1D CNN)
#    Models the sequence of movements + Explicitly flags U-Turns.
# ------------------------------------------------------------------------------
class TemporalHeadingExtractor(nn.Module):
    def __init__(self, output_dim=16):
        super().__init__()
        
        # Input features per step: 3 (Yaw Rate, Velocity X, Velocity Y)
        input_channels = 3 
        
        # Temporal Encoder (1D CNN)
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Pool across time -> [Batch, 32]
        )
        
        # Final Projection (32 features + 1 U-Turn flag -> output_dim)
        self.proj = nn.Sequential(
            nn.Linear(32 + 1, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, positions, padding_mask):
        """
        positions: [Batch, Time, 2]
        """
        # --- A. Pre-process Geometry (Time Series) ---
        vel = positions[:, 1:] - positions[:, :-1]
        
        yaws = torch.atan2(vel[..., 1], vel[..., 0])
        yaw_diff = yaws[:, 1:] - yaws[:, :-1]
        # Normalize angles to range (-pi, pi)
        yaw_diff = (yaw_diff + torch.pi) % (2 * torch.pi) - torch.pi
        
        vel_aligned = vel[:, 1:] 
        
        # Stack features: [Batch, T-2, 3]
        feats = torch.cat([yaw_diff.unsqueeze(-1), vel_aligned], dim=-1)
        
        # Mask out padding
        valid_mask = ~padding_mask[:, 2:]
        feats = feats * valid_mask.unsqueeze(-1).float()
        
        # --- B. Temporal Convolution ---
        feats_permuted = feats.permute(0, 2, 1) # [Batch, Channels, Time]
        temporal_embed = self.conv_net(feats_permuted).squeeze(-1)
        
        # --- C. The "Cheat Code" (Relaxed Threshold) ---
        total_turn = yaw_diff.sum(dim=1, keepdim=True)
        
        # CHANGED: Threshold lowered to 2.5 rad (~143 degrees) to catch wider U-turns
        is_uturn = (total_turn.abs() > 2.5).float()
        
        # --- D. Final Concatenation ---
        out = torch.cat([temporal_embed, is_uturn], dim=1)
        
        return self.proj(out)


# ------------------------------------------------------------------------------
# 3. MAP ENCODER
# ------------------------------------------------------------------------------
class MapEncoder(nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()
        self.turn_embed = nn.Embedding(3, 8)       
        self.intersect_embed = nn.Embedding(2, 4)  
        self.control_embed = nn.Embedding(2, 4)    
        self.net = nn.Sequential(
            nn.Linear(16, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, turn, intersect, control):
        t = self.turn_embed(turn)
        i = self.intersect_embed(intersect)
        c = self.control_embed(control)
        x = torch.cat([t, i, c], dim=-1)
        return self.net(x)


# ------------------------------------------------------------------------------
# 4. MAIN CLASSIFIER
# ------------------------------------------------------------------------------
class ManeuverClassifier(pl.LightningModule):
    def __init__(
        self,
        frozen_backbone,
        num_classes: int = 7,
        learning_rate: float = 1e-3,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["frozen_backbone"])

        # A. ENCODERS
        self.backbone = frozen_backbone
        self.map_encoder = MapEncoder(output_dim=32)
        self.geo_extractor = TemporalHeadingExtractor(output_dim=16)

        # B. CLASSIFICATION HEAD
        # Input size: 128 (Backbone) + 32 (Map) + 16 (Geometry) = 176
        self.head = nn.Sequential(
            nn.Linear(176, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, num_classes),
        )

        # C. LOSS & METRICS (UPDATED TO FOCAL LOSS)
        self.criterion = FocalLoss(alpha=None, gamma=2.0)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
        
        self.class_names = ["Straight", "Left Turn", "Right Turn", "U-Turn", "LC Left", "LC Right", "Stationary"]

    def forward(self, batch):
        # 1. TRAJECTORY EMBEDDING (Backbone)
        rotate = getattr(self.backbone.hparams, 'rotate', True)
        if rotate:
            rotate_mat = torch.empty(batch.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(batch['rotate_angles'])
            cos_vals = torch.cos(batch['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            batch['rotate_mat'] = rotate_mat
        else:
            batch['rotate_mat'] = None

        local_embed = self.backbone.local_encoder(data=batch)
        out = self.backbone.global_interactor(data=batch, local_embed=local_embed)
        
        ego_idx = batch.ego_index.long()
        traj_embed = out[0, ego_idx, :] # [Batch, 128]

        # 2. MAP EMBEDDING
        turn = batch.turn_directions[ego_idx] if hasattr(batch, 'turn_directions') else torch.zeros_like(ego_idx)
        intersect = batch.is_intersections[ego_idx] if hasattr(batch, 'is_intersections') else torch.zeros_like(ego_idx)
        control = batch.traffic_controls[ego_idx] if hasattr(batch, 'traffic_controls') else torch.zeros_like(ego_idx)
        
        map_embed = self.map_encoder(
            torch.clamp(turn, 0, 2).long(),
            torch.clamp(intersect, 0, 1).long(),
            torch.clamp(control, 0, 1).long()
        ) # [Batch, 32]

        # 3. GEOMETRY EMBEDDING
        ego_pos = batch['positions'][ego_idx]
        ego_mask = batch['padding_mask'][ego_idx]
        geo_embed = self.geo_extractor(ego_pos, ego_mask) # [Batch, 16]

        # 4. FUSION
        fused = torch.cat([traj_embed, map_embed, geo_embed], dim=1)
        
        return self.head(fused)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': 1e-5},
            {'params': self.map_encoder.parameters(), 'lr': 1e-3},
            {'params': self.geo_extractor.parameters(), 'lr': 1e-3},
            {'params': self.head.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def training_step(self, batch, batch_idx):
        targets = batch.maneuver_id.view(-1).long()
        logits = self(batch)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, targets)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.maneuver_id.view(-1)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, targets)
        self.val_f1_per_class.update(preds, targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=targets.size(0))
        return loss

    def on_validation_epoch_end(self):
        f1_scores = self.val_f1_per_class.compute()
        acc = self.val_acc.compute()
        macro_f1 = f1_scores.mean()
        self.log("val_f1_macro", macro_f1, prog_bar=True)
        self.log("val_acc_epoch", acc, prog_bar=False)
        if self.trainer.is_global_zero:
            print(f"\n{'='*60}")
            print(f"Epoch {self.current_epoch} Results | Macro F1: {macro_f1:.4f} | Acc: {acc:.2%}")
            print(f"{'-'*60}")
            print(f"{'Class':<15} | {'F1 Score':<10} | {'Status'}")
            print(f"{'-'*60}")
            for i, name in enumerate(self.class_names):
                score = f1_scores[i].item()
                status = "⚠️ LOW" if score < 0.5 else "✅ OK"
                print(f"{name:<15} | {score:.4f}     | {status}")
            print(f"{'='*60}\n")
        self.val_f1_per_class.reset()
        self.val_acc.reset()