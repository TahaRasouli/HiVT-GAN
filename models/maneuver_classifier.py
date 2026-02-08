import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score

class ManeuverClassifier(pl.LightningModule):
    def __init__(self, frozen_backbone, num_classes=6, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['frozen_backbone'])
        self.lr = lr
        self.encoder = frozen_backbone.local_encoder
        self.interactor = frozen_backbone.global_interactor
        
        # Linear Head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.register_buffer("loss_weights", class_weights)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.class_names = ["Straight", "Left", "Right", "LCL", "LCR", "Stat"]

    def forward(self, data):
        # 1. On-the-fly Rotation Matrix
        rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
        sin, cos = torch.sin(data['rotate_angles']), torch.cos(data['rotate_angles'])
        rotate_mat[:, 0, 0] = cos; rotate_mat[:, 0, 1] = -sin
        rotate_mat[:, 1, 0] = sin; rotate_mat[:, 1, 1] = cos
        data['rotate_mat'] = rotate_mat
        
        # 2. Backbone Forward
        local_embed = self.encoder(data=data)
        global_embed = self.interactor(data=data, local_embed=local_embed)
        
        # 3. Vectorized Ego Extraction
        # Egos are the first node of each graph in the batch
        indices = torch.cat([torch.tensor([0], device=self.device), 
                             torch.where(data.batch[1:] != data.batch[:-1])[0] + 1])
        
        return self.head(global_embed[indices])

    def _remap(self, y):
        mapping = torch.full_like(y, -1)

        mapping[y == 0] = 0
        mapping[y == 1] = 1
        mapping[y == 2] = 2
        mapping[y == 4] = 3
        mapping[y == 5] = 4
        mapping[y == 6] = 5

        if (mapping < 0).any():
            bad = torch.unique(y[mapping < 0])
            raise RuntimeError(f"Invalid maneuver labels found: {bad}")

        return mapping


    def training_step(self, data, batch_idx):
        logits = self(data)
        indices = torch.cat([
            torch.tensor([0], device=self.device),
            torch.where(data.batch[1:] != data.batch[:-1])[0] + 1
        ])

        y = self._remap(data.maneuver_id[indices])

        loss = F.cross_entropy(logits, y, weight=self.loss_weights)
        self.log("train_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        indices = torch.cat([
            torch.tensor([0], device=self.device),
            torch.where(data.batch[1:] != data.batch[:-1])[0] + 1
        ])

        y = self._remap(data.maneuver_id[indices])
        print("logits shape:", logits.shape)
        print("y shape:", y.shape)
        print("y unique:", torch.unique(y))
        self.val_f1.update(logits, y)
        return F.cross_entropy(logits, y, weight=self.loss_weights)

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        if self.trainer.is_global_zero:
            print(f"\nEpoch {self.current_epoch} | Macro F1: {f1:.4f}")
        self.log("val_f1_macro", f1, prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)