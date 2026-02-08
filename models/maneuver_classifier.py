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
            """
            Efficiently remaps non-contiguous labels to [0, 5].
            0:Straight, 1:Left, 2:Right, 4:LCL, 5:LCR, 6:Stat
            """
            # Create a copy to avoid in-place modification of the original data
            new_y = torch.zeros_like(y)
            
            # Vectorized remapping (No .item() or list comps - safe for CUDA)
            new_y = torch.where(y == 1, torch.tensor(1, device=y.device), new_y) # Left
            new_y = torch.where(y == 2, torch.tensor(2, device=y.device), new_y) # Right
            new_y = torch.where(y == 4, torch.tensor(3, device=y.device), new_y) # LCL -> 3
            new_y = torch.where(y == 5, torch.tensor(4, device=y.device), new_y) # LCR -> 4
            new_y = torch.where(y == 6, torch.tensor(5, device=y.device), new_y) # Stat -> 5
            # y=0 stays new_y=0 (Straight)
            
            return new_y

    def training_step(self, data, batch_idx):
        logits = self(data)
        y = self._remap(data.maneuver_id.view(-1))
        loss = F.cross_entropy(logits, y, weight=self.loss_weights)
        self.log("train_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        y = self._remap(data.maneuver_id.view(-1))
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