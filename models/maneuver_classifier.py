import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score

class ManeuverClassifier(pl.LightningModule):
    def __init__(self, frozen_backbone, num_classes=6, lr=5e-4, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['frozen_backbone'])
        self.lr = lr
        
        # 1. Access components of the frozen CVAE
        self.encoder = frozen_backbone.local_encoder
        self.interactor = frozen_backbone.global_interactor
        
        # 2. Trajectory Projection (Encodes the future coordinates)
        # 30 steps * 2 (x,y) = 60 input features
        self.traj_projection = nn.Sequential(
            nn.Linear(60, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 3. Classification Head
        # Fuses Context (128) + Trajectory (64) = 192
        self.head = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.register_buffer("loss_weights", class_weights)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.class_names = ["Straight", "Left", "Right", "LCL", "LCR", "Stat"]

    def forward(self, data, future_traj):
        """
        data: Batch of graphs
        future_traj: [Batch, 30, 2]
        """
        # A. Context Extraction (Backbone logic)
        rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
        sin, cos = torch.sin(data.rotate_angles), torch.cos(data.rotate_angles)
        rotate_mat[:,0,0] = cos; rotate_mat[:,0,1] = -sin
        rotate_mat[:,1,0] = sin; rotate_mat[:,1,1] = cos
        data.rotate_mat = rotate_mat

        local_embed = self.encoder(data=data)
        global_embed = self.interactor(data=data, local_embed=local_embed)

        # Get Ego indices (Index 0 of each graph in batch)
        batch = data.batch
        ego_indices = torch.cat([
            torch.tensor([0], device=batch.device),
            torch.where(batch[1:] != batch[:-1])[0] + 1
        ])
        ego_context = global_embed[ego_indices] # [Batch, 128]

        # B. Trajectory Encoding
        # Reshape [B, 30, 2] -> [B, 60]
        traj_feat = self.traj_projection(future_traj.reshape(future_traj.size(0), -1))

        # C. Fusion & Prediction
        fused = torch.cat([ego_context, traj_feat], dim=-1) # [Batch, 192]
        return self.head(fused)

    def training_step(self, data, batch_idx):
        # During training, we only look at the Ground Truth (data.y)
        logits = self(data, data.y)
        y = self._remap(data.maneuver_id)
        
        loss = F.cross_entropy(logits, y, weight=self.loss_weights)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _remap(self, y):
        # 0:Straight, 1:Left, 2:Right, 4:LCL, 5:LCR, 6:Stat -> 0,1,2,3,4,5
        mapping = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5}
        new_y = torch.zeros_like(y)
        for old, new in mapping.items():
            new_y[y == old] = new
        return new_y

    def validation_step(self, data, batch_idx):
        logits = self(data, data.y)
        y = self._remap(data.maneuver_id)
        self.val_f1_macro.update(logits, y)
        return F.cross_entropy(logits, y, weight=self.loss_weights)

    def on_validation_epoch_end(self):
        f1 = self.val_f1_macro.compute()
        self.log("val_f1_macro", f1, prog_bar=True)
        self.val_f1_macro.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)