import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import F1Score
from torch_geometric.nn import global_mean_pool

class ManeuverClassifier(pl.LightningModule):
    def __init__(self, frozen_backbone, num_classes=6, lr=5e-4, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['frozen_backbone'])

        # 1. Encoder (frozen)
        self.encoder = frozen_backbone
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 2. Classifier head
        embed_dim = self.encoder.hparams.embed_dim  # Must match CVAE output embedding
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # 3. Loss function
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # 4. Metrics
        self.train_f1 = F1Score(num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(num_classes=num_classes, average='macro')

        # For per-class F1
        self.val_preds = []
        self.val_targets = []

        self.lr = lr

    def forward(self, batch):
        # Encode graph nodes
        node_features = self.encoder(batch)  # [num_nodes, embed_dim]

        # Pool node features per graph
        graph_features = global_mean_pool(node_features, batch.batch)  # [num_graphs, embed_dim]

        # Pass through classifier
        logits = self.classifier(graph_features)  # [num_graphs, num_classes]
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.maneuver_id.view(-1)
        loss = self.loss_fn(logits, targets)

        # Track training F1
        preds = torch.argmax(logits, dim=-1)
        self.train_f1.update(preds, targets)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def training_epoch_end(self, outputs):
        f1 = self.train_f1.compute()
        self.log("train_f1_macro", f1, prog_bar=True)
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.maneuver_id.view(-1)
        loss = self.loss_fn(logits, targets)

        preds = torch.argmax(logits, dim=-1)
        self.val_preds.append(preds)
        self.val_targets.append(targets)

        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        # Compute per-class F1
        per_class_f1 = []
        for cls in range(self.hparams.num_classes):
            cls_mask = targets == cls
            if cls_mask.sum() == 0:
                f1 = torch.tensor(0.0, device=self.device)
            else:
                cls_preds = preds[cls_mask]
                cls_targets = targets[cls_mask]
                f1 = F1Score(num_classes=1, average='macro')(cls_preds, cls_targets)
            per_class_f1.append(f1.item())

        macro_f1 = torch.tensor(per_class_f1).mean().item()
        print(f"\nEpoch {self.current_epoch} | val_macro_f1: {macro_f1:.4f} | per-class F1: {per_class_f1}")

        # Reset lists for next epoch
        self.val_preds.clear()
        self.val_targets.clear()

        # Log for Trainer
        self.log("val_f1_macro", macro_f1, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
