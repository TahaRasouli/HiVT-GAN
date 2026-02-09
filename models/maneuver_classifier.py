import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.functional.classification import multiclass_f1_score


class ManeuverClassifier(pl.LightningModule):

    def __init__(self, frozen_backbone, num_classes=6, lr=1e-3, class_weights=None):
        super().__init__()

        # DO NOT save backbone modules inside hyperparams
        self.save_hyperparameters(ignore=["frozen_backbone"])

        self.lr = lr

        # ----- Backbone -----
        self.encoder = frozen_backbone  # CVAE model

        # ----- Classification head -----
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        # ----- Loss -----
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # ----- Metrics storage -----
        self.val_preds = []
        self.val_targets = []

        self.class_names = [
            "Straight",
            "Left",
            "Right",
            "LCL",
            "LCR",
            "Stat",
        ]

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, batch):

        # backbone returns [1, num_nodes, embed_dim]
        node_features = self.encoder(batch)

        # remove mode dimension
        node_features = node_features.squeeze(0)   # [num_nodes, 128]

        # aggregate node features → graph features
        graph_features = torch.zeros(
            batch.num_graphs,
            node_features.size(-1),
            device=node_features.device,
        )

        graph_features = graph_features.index_add(
            0,
            batch.batch,
            node_features,
        )

        logits = self.classifier(graph_features)

        return logits

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def training_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        loss = self.loss_fn(logits, targets)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        return loss

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validation_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        preds = torch.argmax(logits, dim=-1)

        # store for epoch metrics
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    # ---------------------------------------------------------
    # Per-class F1 after epoch
    # ---------------------------------------------------------

    @torch.no_grad()
    def on_validation_epoch_end(self):

        if len(self.val_preds) == 0:
            return

        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)

        per_class_f1 = multiclass_f1_score(
            preds,
            targets,
            num_classes=6,
            average=None,
        )

        macro_f1 = per_class_f1.mean()

        print("\n========== Validation F1 ==========")

        for i, f1 in enumerate(per_class_f1):
            print(f"{self.class_names[i]:<10} F1: {f1.item():.4f}")

        print(f"Macro F1: {macro_f1.item():.4f}")
        print("===================================")

        self.log("val_f1_macro", macro_f1, prog_bar=True)

        # reset storage
        self.val_preds.clear()
        self.val_targets.clear()

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    def configure_optimizers(self):

        return torch.optim.AdamW(self.parameters(), lr=self.lr)
