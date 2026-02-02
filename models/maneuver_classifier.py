import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

# ------------------------------------------------------------------------------
# 1. MAP ENCODER
#    Learns a compact representation of categorical map features.
# ------------------------------------------------------------------------------
class MapEncoder(nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()
        # Tiny embedding tables for categorical inputs
        # Turn Direction: 0=Straight, 1=Left, 2=Right
        self.turn_embed = nn.Embedding(3, 8)       
        
        # Is Intersection: 0=No, 1=Yes
        self.intersect_embed = nn.Embedding(2, 4)  
        
        # Traffic Control: 0=No, 1=Yes
        self.control_embed = nn.Embedding(2, 4)    
        
        # Simple projection to mix the features into a single vector
        # Input size: 8 (Turn) + 4 (Intersect) + 4 (Control) = 16
        self.net = nn.Sequential(
            nn.Linear(16, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, turn, intersect, control):
        t = self.turn_embed(turn)
        i = self.intersect_embed(intersect)
        c = self.control_embed(control)
        
        # Concatenate features and project
        x = torch.cat([t, i, c], dim=-1)
        return self.net(x)


# ------------------------------------------------------------------------------
# 2. GATED FUSION MODULE
#    Dynamically controls the flow of Map Information based on Trajectory context.
# ------------------------------------------------------------------------------
class GatedFusion(nn.Module):
    def __init__(self, traj_dim=128, map_dim=32, out_dim=128):
        super().__init__()
        # Projects map to same size as trajectory for element-wise operations
        self.map_proj = nn.Linear(map_dim, traj_dim)
        
        # The Gate: Looks at BOTH Traj and Map to decide importance
        # Output is 1 value per channel (sigmoid -> 0.0 to 1.0)
        self.gate_net = nn.Sequential(
            nn.Linear(traj_dim + map_dim, traj_dim),
            nn.Sigmoid()
        )
        
        # Final processing layer after fusion
        self.out_net = nn.Sequential(
            nn.Linear(traj_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, traj_embed, map_embed):
        # 1. Project Map to Trajectory dimension [B, 128]
        map_feat = self.map_proj(map_embed)
        
        # 2. Calculate Gate (Importance of Map context)
        # We concatenate raw inputs to decide the gate
        combined_raw = torch.cat([traj_embed, map_embed], dim=1)
        gate = self.gate_net(combined_raw) # [B, 128] (Values 0.0 to 1.0)
        
        # 3. Apply Gate: Trajectory + (Gate * Map)
        # If Gate is near 0, we ignore Map. If Gate is near 1, we use full Map.
        fused = traj_embed + (gate * map_feat)
        
        return self.out_net(fused)


# ------------------------------------------------------------------------------
# 3. MAIN CLASSIFIER MODULE
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

        # A. BACKBONE (Trajectory Encoder)
        self.backbone = frozen_backbone

        # B. MAP ENCODER
        self.map_encoder = MapEncoder(output_dim=32)

        # C. FUSION MODULE (Gated)
        self.fusion = GatedFusion(traj_dim=128, map_dim=32, out_dim=128)

        # D. CLASSIFICATION HEAD
        # Input: 128 (Fused Vector)
        self.head = nn.Sequential(
            nn.Dropout(0.5), # Regularization
            nn.Linear(128, num_classes),
        )

        # E. METRICS & LOSS
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

        self.class_names = [
            "Straight", "Left Turn", "Right Turn", "U-Turn",
            "LC Left", "LC Right", "Stationary",
        ]

    def forward(self, batch):
        # ----------------------------------------------------------
        # 1. TRAJECTORY ENCODING (Backbone Bypass)
        # ----------------------------------------------------------
        # Prepare Rotation Matrix
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

        # Run Backbone Encoders Manually
        local_embed = self.backbone.local_encoder(data=batch)
        out = self.backbone.global_interactor(data=batch, local_embed=local_embed)
        
        # Extract Ego Embedding [Batch, 128]
        if not torch.is_tensor(out):
             raise RuntimeError(f"Backbone encoders returned {type(out)} instead of Tensor")
        ego_idx = batch.ego_index.long()
        traj_embed = out[0, ego_idx, :] 

        # ----------------------------------------------------------
        # 2. MAP ENCODING
        # ----------------------------------------------------------
        # Fetch features, defaulting to 0 if missing
        turn = batch.turn_directions[ego_idx] if hasattr(batch, 'turn_directions') else torch.zeros_like(ego_idx)
        intersect = batch.is_intersections[ego_idx] if hasattr(batch, 'is_intersections') else torch.zeros_like(ego_idx)
        control = batch.traffic_controls[ego_idx] if hasattr(batch, 'traffic_controls') else torch.zeros_like(ego_idx)

        # Clamp to valid ranges for embedding lookup
        turn = torch.clamp(turn, 0, 2).long()
        intersect = torch.clamp(intersect, 0, 1).long()
        control = torch.clamp(control, 0, 1).long()

        # Generate Map Embedding [Batch, 32]
        map_embed = self.map_encoder(turn, intersect, control)

        # ----------------------------------------------------------
        # 3. GATED FUSION
        # ----------------------------------------------------------
        # Combine [128] + [32] -> [128]
        fused_features = self.fusion(traj_embed, map_embed)

        # ----------------------------------------------------------
        # 4. CLASSIFICATION
        # ----------------------------------------------------------
        logits = self.head(fused_features)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': 1e-5},    # Slow Backbone
            {'params': self.map_encoder.parameters(), 'lr': 1e-3}, # Fast Map Encoder
            {'params': self.fusion.parameters(), 'lr': 1e-3},      # Fast Fusion
            {'params': self.head.parameters(), 'lr': 1e-3}         # Fast Head
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

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