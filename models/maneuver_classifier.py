import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

# ------------------------------------------------------------------------------
# 1. HEADING & GEOMETRY EXTRACTOR (NEW)
#    Explicitly calculates geometry that CNNs/Transformers struggle to infer.
# ------------------------------------------------------------------------------
class HeadingExtractor(nn.Module):
    def __init__(self, output_dim=16):
        super().__init__()
        # We process the raw geometric stats into a feature vector
        self.net = nn.Sequential(
            nn.Linear(4, output_dim), # Input: [Total Turn, Max Rate, Avg Rate, Displacement]
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, positions, padding_mask):
        """
        positions: [Batch, Time, 2] (Global coordinates)
        padding_mask: [Batch, Time] (True = padding)
        """
        # 1. Calculate Velocity Vectors
        # shape: [B, T-1, 2]
        vel = positions[:, 1:] - positions[:, :-1]
        
        # 2. Calculate Yaw (Heading) for each step
        # atan2 gives angle in radians (-pi to pi)
        yaws = torch.atan2(vel[..., 1], vel[..., 0])
        
        # 3. Calculate Yaw Rate (Change in Heading)
        # shape: [B, T-2]
        yaw_diff = yaws[:, 1:] - yaws[:, :-1]
        
        # 4. Handle Wraparound (e.g. 179 deg -> -179 deg is a small turn, not huge)
        yaw_diff = (yaw_diff + torch.pi) % (2 * torch.pi) - torch.pi
        
        # 5. Filter Padding (Zero out invalid steps)
        # We use the mask from the backbone data
        valid_mask = ~padding_mask[:, 2:] # Align mask with yaw_diff size
        yaw_diff = yaw_diff * valid_mask.float()

        # 6. Extract Statistical Features
        # A. Total Heading Change (Integral of curvature) - distinguishes Straight vs Turn
        total_turn = yaw_diff.sum(dim=1, keepdim=True)
        
        # B. Max Yaw Rate - distinguishes Sharp Turn vs Wide Curve
        max_rate = yaw_diff.abs().max(dim=1, keepdim=True)[0]
        
        # C. Average Yaw Rate - General curviness
        # Avoid division by zero
        steps = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        avg_rate = yaw_diff.abs().sum(dim=1, keepdim=True) / steps
        
        # D. Displacement Angle (End Point vs Start Point)
        # This captures the "Net" result of the maneuver
        start_pos = positions[:, 0]
        # Find last valid position
        last_indices = valid_mask.long().sum(dim=1) 
        # (Simplified gathering for last valid pos, usually just use last index for fixed horizon)
        end_pos = positions[:, -1] 
        disp_vec = end_pos - start_pos
        net_heading = torch.atan2(disp_vec[:, 1], disp_vec[:, 0]).unsqueeze(1)

        # Concat geometric features [Batch, 4]
        feats = torch.cat([total_turn, max_rate, avg_rate, net_heading], dim=1)
        
        # Project to embedding size
        return self.net(feats)

# ------------------------------------------------------------------------------
# 2. MAP ENCODER (Unchanged)
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
# 3. MAIN CLASSIFIER (Direct Fusion)
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
        self.geo_extractor = HeadingExtractor(output_dim=16)

        # B. CLASSIFICATION HEAD
        # Input size calculation:
        # 128 (Backbone) + 32 (Map) + 16 (Geometry) = 176
        self.head = nn.Sequential(
            nn.Linear(176, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # High dropout essential for fusion
            nn.Linear(128, num_classes),
        )

        # C. METRICS
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
        
        self.class_names = ["Straight", "Left Turn", "Right Turn", "U-Turn", "LC Left", "LC Right", "Stationary"]

    def forward(self, batch):
        # 1. TRAJECTORY EMBEDDING (Backbone) -----------------------
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

        # 2. MAP EMBEDDING (Always Considered) ---------------------
        turn = batch.turn_directions[ego_idx] if hasattr(batch, 'turn_directions') else torch.zeros_like(ego_idx)
        intersect = batch.is_intersections[ego_idx] if hasattr(batch, 'is_intersections') else torch.zeros_like(ego_idx)
        control = batch.traffic_controls[ego_idx] if hasattr(batch, 'traffic_controls') else torch.zeros_like(ego_idx)
        
        map_embed = self.map_encoder(
            torch.clamp(turn, 0, 2).long(),
            torch.clamp(intersect, 0, 1).long(),
            torch.clamp(control, 0, 1).long()
        ) # [Batch, 32]

        # 3. GEOMETRY EMBEDDING (Heading Changes) ------------------
        # We need raw positions for the Ego agent
        # batch['positions'] is [Total_Nodes, Time, 2]
        ego_pos = batch['positions'][ego_idx] # [Batch, Time, 2]
        ego_mask = batch['padding_mask'][ego_idx] # [Batch, Time]
        
        geo_embed = self.geo_extractor(ego_pos, ego_mask) # [Batch, 16]

        # 4. DIRECT FUSION -----------------------------------------
        # Concatenate everything. The MLP handles mixing.
        # Size: 128 + 32 + 16 = 176
        fused = torch.cat([traj_embed, map_embed, geo_embed], dim=1)
        
        return self.head(fused)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': 1e-5},
            {'params': self.map_encoder.parameters(), 'lr': 1e-3},
            {'params': self.geo_extractor.parameters(), 'lr': 1e-3}, # New params
            {'params': self.head.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4) # Slight weight decay helps fusion
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    # Copy paste training_step, validation_step, on_validation_epoch_end from previous code
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