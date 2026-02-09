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
        embed_dim = self.encoder.hparams.embed_dim
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

        # 4. Metrics (fixed with task="multiclass")
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.val_preds = []
        self.val_targets = []

        self.lr = lr

    def forward(self, batch):
        node_features = self.encoder(batch)
        print("node_features shape:", node_features.shape)
        print("batch.batch shape:", getattr(batch, 'batch', None))
        graph_features = global_mean_pool(node_features, batch.batch)
        print("graph_features shape:", graph_features.shape)
        logits = self.classifier(graph_features)
        print("logits shape:", logits.shape)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.maneuver_id.view(-1)
        loss = self.loss_fn(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.train_f1.update(preds, targets)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

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
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)
        val_f1_score = self.val_f1(preds, targets)
        self.log("val_f1_macro", val_f1_score, prog_bar=True)
        self.val_preds.clear()
        self.val_targets.clear()
        print(f"\nEpoch {self.current_epoch} | val_f1_macro: {val_f1_score:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
