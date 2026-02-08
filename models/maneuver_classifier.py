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
        
        # Backbone setup
        self.encoder = frozen_backbone.local_encoder
        self.interactor = frozen_backbone.global_interactor
        
        # Classification Head (6 classes: 0, 1, 2, 4, 5, 6)
        # Note: We skip ID 3 (U-Turn)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self.register_buffer("loss_weights", class_weights if class_weights is not None else torch.ones(num_classes))
        
        # Metrics (F1 is best for the remaining imbalance)
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        
        # Labels for the report (excluding U-Turn)
        self.class_names = ["Straight", "Left Turn", "Right Turn", "Lane Chg L", "Lane Chg R", "Stationary"]

    def forward(self, data):
        local_embed = self.encoder(data=data)
        global_embed = self.interactor(data=data, local_embed=local_embed)
        ego_embeddings = global_embed[data.ptr[:-1]] 
        return self.head(ego_embeddings)

    def training_step(self, data, batch_idx):
        y_hat = self(data)
        y = self._remap_labels(data.maneuver_id.view(-1))
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weights)
        self.log("train_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def _remap_labels(self, y):
        """Remaps 0,1,2,4,5,6 to 0,1,2,3,4,5 to keep indices contiguous."""
        mapping = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5}
        return torch.tensor([mapping[val.item()] for val in y], device=y.device)

    def validation_step(self, data, batch_idx):
        y_hat = self(data)
        y = self._remap_labels(data.maneuver_id.view(-1))
        
        self.val_f1_per_class.update(y_hat, y)
        self.val_acc.update(y_hat, y)
        return F.cross_entropy(y_hat, y)

    def on_validation_epoch_end(self):
        f1_scores = self.val_f1_per_class.compute()
        macro_f1 = f1_scores.mean()
        
        if self.trainer.is_global_zero:
            print(f"\n--- Epoch {self.current_epoch} Maneuver Report ---")
            for i, name in enumerate(self.class_names):
                print(f"{name:<15}: F1={f1_scores[i]:.4f}")
            print(f"Total Macro F1: {macro_f1:.4f}\n")
            
        self.log("val_f1_macro", macro_f1, prog_bar=True)
        self.val_f1_per_class.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)