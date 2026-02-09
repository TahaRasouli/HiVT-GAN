import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score

class ManeuverClassifier(pl.LightningModule):
    def __init__(self, encoder, embed_dim=128, num_classes=6, lr=5e-4, loss_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])  # <<--- IGNORE encoder

        self.encoder = encoder
        self.lr = lr
        self.num_classes = num_classes

        # Classification head: fuse trajectory embedding + optional map encoding
        # If map features are already encoded, we just concatenate: [traj_embed + map_embed]
        # Here, for simplicity, we assume map features same dim as embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes)
        )

        if loss_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data, map_encoding=None):
        """
        Args:
            data: PyG batch of agent trajectories
            map_encoding: tensor [num_agents, embed_dim] precomputed map features
        Returns:
            logits: [num_agents, num_classes]
        """
        traj_embed = self.encoder(data)  # [num_agents, embed_dim]
        if map_encoding is not None:
            x = torch.cat([traj_embed, map_encoding], dim=-1)  # [num_agents, embed*2]
        else:
            # If map_encoding missing, duplicate trajectory embed
            x = torch.cat([traj_embed, traj_embed], dim=-1)

        logits = self.classifier(x)
        return logits

    # -------------------------
    # Training Step
    # -------------------------
    def training_step(self, batch, batch_idx):
        labels = batch.maneuver_id
        logits = self(batch, map_encoding=getattr(batch, 'map_encoding', None))
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------
    # Validation Step
    # -------------------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        labels = batch.maneuver_id
        logits = self(batch, map_encoding=getattr(batch, 'map_encoding', None))
        preds = torch.argmax(logits, dim=-1)

        # Store for epoch-level F1
        if not hasattr(self, "val_preds"):
            self.val_preds = []
            self.val_labels = []
        self.val_preds.append(preds.cpu())
        self.val_labels.append(labels.cpu())

    # -------------------------
    # Validation Epoch End
    # -------------------------
    @torch.no_grad()
    def on_validation_epoch_end(self):
        if hasattr(self, "val_preds"):
            preds = torch.cat(self.val_preds, dim=0)
            labels = torch.cat(self.val_labels, dim=0)

            # F1 per class
            f1_per_class = f1_score(labels.numpy(), preds.numpy(), average=None, zero_division=0)
            f1_macro = f1_score(labels.numpy(), preds.numpy(), average="macro")

            # Log
            for i, f1c in enumerate(f1_per_class):
                self.log(f"val_f1_class_{i}", f1c, prog_bar=True)
            self.log("val_f1_macro", f1_macro, prog_bar=True)

            print(f"\nEpoch {self.current_epoch:03d} | Val F1 Macro: {f1_macro:.4f} | per class: {f1_per_class}")

            # Reset
            self.val_preds = []
            self.val_labels = []

    # -------------------------
    # Optimizer
    # -------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
