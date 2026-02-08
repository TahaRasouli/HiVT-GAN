import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

class ManeuverClassifier(pl.LightningModule):
    def __init__(self, frozen_backbone, num_classes=6, embed_dim=128, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['frozen_backbone'])
        self.lr = lr
        
        # Access components of the frozen backbone
        self.encoder = frozen_backbone.local_encoder
        self.interactor = frozen_backbone.global_interactor
        
        # Classification Head (mapping to 6 classes)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self.register_buffer("loss_weights", class_weights if class_weights is not None else torch.ones(num_classes))
        
        # Metrics
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.class_names = ["Straight", "Left Turn", "Right Turn", "Lane Chg L", "Lane Chg R", "Stationary"]

    def forward(self, data):
        # --- CRITICAL FIX: Calculate rotate_mat on the fly ---
        # This part was missing and caused the KeyError in LocalEncoder
        rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
        sin_vals = torch.sin(data['rotate_angles'])
        cos_vals = torch.cos(data['rotate_angles'])
        rotate_mat[:, 0, 0] = cos_vals; rotate_mat[:, 0, 1] = -sin_vals
        rotate_mat[:, 1, 0] = sin_vals; rotate_mat[:, 1, 1] = cos_vals
        
        # Attach it to the data object temporarily for the encoder to use
        data['rotate_mat'] = rotate_mat
        
        # Now the encoder won't crash
        local_embed = self.encoder(data=data)
        global_embed = self.interactor(data=data, local_embed=local_embed)
        
        # Extract ego embeddings (index 0 of each graph)
        ego_embeddings = global_embed[data.ptr[:-1]] 
        return self.head(ego_embeddings)

    def _remap_labels(self, y):
        # We only have 6 valid classes now
        mapping = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
        
        # Use .get() with a default of 0 to prevent KeyError, 
        # but the dataset filter should have already removed bad IDs.
        remapped = torch.tensor([mapping.get(val.item(), 0) for val in y], device=y.device)
        return remapped

    def training_step(self, data, batch_idx):
        logits = self(data)
        y = self._remap_labels(data.maneuver_id.view(-1))
        loss = F.cross_entropy(logits, y, weight=self.loss_weights)
        self.log("train_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        y = self._remap_labels(data.maneuver_id.view(-1))
        loss = F.cross_entropy(logits, y, weight=self.loss_weights)
        
        self.val_f1_per_class.update(logits, y)
        self.val_acc.update(logits, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def on_validation_epoch_end(self):
        f1_scores = self.val_f1_per_class.compute()
        macro_f1 = f1_scores.mean()
        
        if self.trainer.is_global_zero:
            print(f"\n{'='*65}")
            print(f"Epoch {self.current_epoch} Results | Macro F1: {macro_f1:.4f}")
            print(f"{'-'*65}")
            for i, name in enumerate(self.class_names):
                print(f"{name:<20} | F1: {f1_scores[i]:.4f}")
            print(f"{'='*65}\n")
            
        self.log("val_f1_macro", macro_f1, prog_bar=True)
        self.val_f1_per_class.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)