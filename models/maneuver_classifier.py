import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score


class ManeuverClassifier(pl.LightningModule):

    def __init__(
        self,
        frozen_backbone,
        num_classes=6,
        lr=5e-4,
        class_weights=None,
        id_to_class=None,
        future_steps=30,
    ):
        super().__init__()

        # ------------------------------------------------
        # Backbone (HiVT / CVAE encoder only)
        # ------------------------------------------------
        self.encoder = frozen_backbone

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        embed_dim = self.encoder.hparams.embed_dim

        self.lr = lr
        self.num_classes = num_classes
        self.future_steps = future_steps
        self.id_to_class = id_to_class

        # ------------------------------------------------
        # Trajectory encoder (GROUND TRUTH)
        # ------------------------------------------------
        self.traj_encoder = nn.Sequential(
            nn.Linear(future_steps * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

        # ------------------------------------------------
        # Fusion classifier
        # ------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # ------------------------------------------------
        # Loss
        # ------------------------------------------------
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # ------------------------------------------------
        # Metrics
        # ------------------------------------------------
        self.val_f1_macro = MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        )

        self.val_f1_per_class = MulticlassF1Score(
            num_classes=num_classes,
            average=None
        )

    # ====================================================
    # Forward
    # ====================================================

    def forward(self, batch):

        # --------------------------------------------------
        # 1. Scene encoding
        # --------------------------------------------------
        node_features = self.encoder(batch)  # [1, N_total, D]
        node_features = node_features.squeeze(0)  # [N_total, D]

        batch_index = batch.batch

        # --------------------------------------------------
        # 2. Extract ego nodes
        # (first node of each graph)
        # --------------------------------------------------
        ego_indices = torch.cat([
            torch.tensor([0], device=batch_index.device),
            torch.where(batch_index[1:] != batch_index[:-1])[0] + 1
        ])

        ego_embed = node_features[ego_indices]  # [B, D]

        # --------------------------------------------------
        # 3. USE GROUND TRUTH TRAJECTORY (CRITICAL FIX)
        # --------------------------------------------------
        ego_traj = batch.y[ego_indices]   # [B, T, 2]

        B = ego_traj.size(0)

        # --------------------------------------------------
        # 4. Encode trajectory
        # --------------------------------------------------
        traj_feat = ego_traj.reshape(B, -1)

        traj_embed = self.traj_encoder(traj_feat)

        # --------------------------------------------------
        # 5. Fuse
        # --------------------------------------------------
        fusion = torch.cat([ego_embed, traj_embed], dim=-1)

        # --------------------------------------------------
        # 6. Classify
        # --------------------------------------------------
        logits = self.classifier(fusion)

        return logits

    # ====================================================
    # Training
    # ====================================================

    def training_step(self, batch, batch_idx):

        logits = self(batch)
        targets = batch.maneuver_id.view(-1)

        loss = self.loss_fn(logits, targets)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    # ====================================================
    # Validation
    # ====================================================

    def validation_step(self, batch, batch_idx):

        logits = self(batch)
        targets = batch.maneuver_id.view(-1)

        preds = torch.argmax(logits, dim=1)

        self.val_f1_macro.update(preds, targets)
        self.val_f1_per_class.update(preds, targets)

    def on_validation_epoch_end(self):

        f1_macro = self.val_f1_macro.compute()
        f1_per_class = self.val_f1_per_class.compute()

        self.log("val_f1_macro", f1_macro, prog_bar=True)

        if self.global_rank == 0:
            print("\n==== Per-class F1 ====")
            for i, f in enumerate(f1_per_class):
                name = self.id_to_class.get(i, str(i)) if self.id_to_class else str(i)
                print(f"{name}: {f.item():.4f}")

        self.val_f1_macro.reset()
        self.val_f1_per_class.reset()

    # ====================================================
    # Optimizer
    # ====================================================

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
