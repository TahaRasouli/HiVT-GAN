import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score

class ManeuverClassifier(pl.LightningModule):
    def __init__(self, frozen_backbone, num_classes=6, lr=5e-4, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['frozen_backbone'])

        # Frozen CVAE backbone
        self.encoder = frozen_backbone
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Simple classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.hparams.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # Loss with class weights
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Storage for validation
        self.val_preds = []
        self.val_labels = []

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, data):
        with torch.no_grad():
            features = self.encoder(data)  # [B, embed_dim]
        if features.dim() > 2:
            features = features.reshape(features.size(0), -1)
        logits = self.classifier(features)
        return logits

    # -----------------------------
    # Training Step
    # -----------------------------
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch.maneuver_id.view(-1)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    # -----------------------------
    # Validation Step
    # -----------------------------
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch.maneuver_id.view(-1)

        loss = self.loss_fn(logits, labels)

        # Store predictions and labels per sample (batch), not per node
        preds = torch.argmax(logits, dim=1)

        # IMPORTANT: detach and move to CPU
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss


    # -----------------------------
    # F1 per class at epoch end
    # -----------------------------
    def on_validation_epoch_end(self):
        if not self.val_preds:
            return

        # Concatenate along batch dimension
        try:
            preds = torch.cat(self.val_preds, dim=0)
            labels = torch.cat(self.val_labels, dim=0)
        except RuntimeError:
            # fallback: flatten each tensor to 1D
            preds = torch.cat([p.view(-1) for p in self.val_preds], dim=0)
            labels = torch.cat([l.view(-1) for l in self.val_labels], dim=0)

        from sklearn.metrics import f1_score

        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

        for idx, f1c in enumerate(f1_per_class):
            self.log(f"val_f1_class{idx}", f1c, prog_bar=True)
        self.log("val_f1_macro", f1_macro, prog_bar=True)

        # Clear for next epoch
        self.val_preds.clear()
        self.val_labels.clear()

        print(f"\nEpoch {self.current_epoch:03d} | val_f1_macro: {f1_macro:.4f} | per-class: {f1_per_class}")


    # -----------------------------
    # Optimizer
    # -----------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
