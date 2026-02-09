import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.classification import MulticlassF1Score


class ManeuverClassifier(pl.LightningModule):

    def __init__(
        self,
        frozen_backbone,
        num_classes=6,
        lr=5e-4,
        class_weights=None,
        num_traj_candidates=6,
        future_steps=30,
    ):
        super().__init__()

        # ------------------------------------------------
        # Backbone (CVAE encoder + decoder)
        # ------------------------------------------------
        self.encoder = frozen_backbone

        # Freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        # ------------------------------------------------
        # Settings
        # ------------------------------------------------
        self.num_classes = num_classes
        self.lr = lr
        self.K = num_traj_candidates
        self.future_steps = future_steps

        embed_dim = self.encoder.hparams.embed_dim

        # ------------------------------------------------
        # Trajectory encoder
        # ------------------------------------------------
        self.traj_encoder = nn.Sequential(
            nn.Linear(future_steps * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

        # ------------------------------------------------
        # Classification head
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

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------

    def forward(self, batch):

        # --------------------------------------------------
        # 1. Scene embedding
        # --------------------------------------------------
        scene_embed = self.encoder(batch)

        if scene_embed.dim() == 3:
            scene_embed = scene_embed.squeeze(1)

        B = scene_embed.size(0)

        # --------------------------------------------------
        # 2. Generate trajectory candidates
        # --------------------------------------------------
        context_expanded = scene_embed.repeat_interleave(self.K, dim=0)

        traj_flat, _ = self.encoder.decoder(context_expanded, y_gt=None)

        # TRUE SHAPE:
        # traj_flat = [B*K, T, 2]

        traj = traj_flat.view(
            B,
            self.K,
            self.future_steps,
            2
        )   # [B,K,T,2]

        # --------------------------------------------------
        # 3. Encode trajectory
        # --------------------------------------------------
        traj_feat = traj.reshape(B, self.K, -1)   # [B,K,T*2]

        traj_embed = self.traj_encoder(traj_feat)   # [B,K,D]

        # --------------------------------------------------
        # 4. Fuse scene + trajectory
        # --------------------------------------------------
        scene_expand = scene_embed.unsqueeze(1).repeat(1, self.K, 1)

        fusion = torch.cat([scene_expand, traj_embed], dim=-1)   # [B,K,2D]

        # --------------------------------------------------
        # 5. Classification
        # --------------------------------------------------
        logits_per_candidate = self.classifier(fusion)   # [B,K,C]

        logits = logits_per_candidate.mean(dim=1)   # [B,C]

        return logits



    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    def training_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        loss = self.loss_fn(logits, targets)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    # ------------------------------------------------
    # Validation
    # ------------------------------------------------

    def validation_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        preds = torch.argmax(logits, dim=1)

        self.val_f1_macro.update(preds, targets)
        self.val_f1_per_class.update(preds, targets)

    # ------------------------------------------------
    # Epoch end
    # ------------------------------------------------

    def on_validation_epoch_end(self):

        f1_macro = self.val_f1_macro.compute()
        f1_per_class = self.val_f1_per_class.compute()

        self.log("val_f1_macro", f1_macro, prog_bar=True)

        if self.global_rank == 0:
            print("\n==== Per-class F1 ====")
            for i, f in enumerate(f1_per_class):
                print(f"Class {i}: {f.item():.4f}")

        self.val_f1_macro.reset()
        self.val_f1_per_class.reset()

    # ------------------------------------------------
    # Optimizer
    # ------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
