import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics.classification import MulticlassPrecision


# =====================================================
# TRAJECTORY ENCODER
# =====================================================

class TrajectoryEncoder(nn.Module):

    def __init__(self, future_steps=30, embed_dim=128):
        super().__init__()

        self.future_steps = future_steps

        self.net = nn.Sequential(
            nn.Linear(future_steps * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, traj):
        """
        traj shape:
            [B, T, 2]
        """

        B = traj.size(0)

        traj_flat = traj.reshape(B, self.future_steps * 2)

        return self.net(traj_flat)


# =====================================================
# ADVANCED EGO-CENTRIC MAP ENCODER
# =====================================================

class EgoCentricMapEncoder(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()

        self.lane_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, batch, ego_embed):

        # lane_vectors shape:
        # [L_total, 2]

        lane_vecs = batch.lane_vectors

        lane_embed = self.lane_mlp(lane_vecs)  # [L,D]

        B = ego_embed.size(0)

        # simplified global lane attention
        lanes = lane_embed.unsqueeze(0).repeat(B, 1, 1)  # [B,L,D]

        ego_query = ego_embed.unsqueeze(1)  # [B,1,D]

        attn_out, _ = self.attn(
            ego_query,
            lanes,
            lanes
        )

        return attn_out.squeeze(1)  # [B,D]


# =====================================================
# MANEUVER CLASSIFIER
# =====================================================

class ManeuverClassifier(pl.LightningModule):

    def __init__(
        self,
        frozen_backbone,
        num_classes=6,
        lr=5e-4,
        class_weights=None,
        future_steps=30,
    ):
        super().__init__()

        self.encoder = frozen_backbone

        # Freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        self.lr = lr
        self.future_steps = future_steps
        self.num_classes = num_classes

        embed_dim = self.encoder.hparams.embed_dim

        # ------------------------------------------------
        # Modules
        # ------------------------------------------------

        self.traj_encoder = TrajectoryEncoder(
            future_steps=future_steps,
            embed_dim=embed_dim
        )

        self.map_encoder = EgoCentricMapEncoder(
            embed_dim=embed_dim
        )

        # Balanced fusion (fix follow collapse)
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # ------------------------------------------------
        # Loss
        # ------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # ------------------------------------------------
        # Metric: PRECISION ONLY (as requested)
        # ------------------------------------------------

        self.val_precision = MulticlassPrecision(
            num_classes=num_classes,
            average="macro"
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, batch):

        # --------------------------------------------------
        # 1. Scene encoding from frozen backbone
        # --------------------------------------------------

        node_features = self.encoder(batch)  # [1, N_total, D]
        node_features = node_features.squeeze(0)

        batch_index = batch.batch

        # Ego index extraction
        ego_indices = torch.cat([
            torch.tensor([0], device=batch_index.device),
            torch.where(batch_index[1:] != batch_index[:-1])[0] + 1
        ])

        ego_embed = node_features[ego_indices]  # [B,D]

        # --------------------------------------------------
        # 2. Trajectory encoding (GROUND TRUTH)
        # --------------------------------------------------

        traj = batch.y[:, :, :]   # ego already index 0 in your data

        traj = traj[:, :, :]  # [B,T,2]

        traj_embed = self.traj_encoder(traj)

        # --------------------------------------------------
        # 3. Map encoding
        # --------------------------------------------------

        map_embed = self.map_encoder(batch, ego_embed)

        # --------------------------------------------------
        # 4. Balanced fusion
        # --------------------------------------------------

        fusion = torch.cat(
            [ego_embed, traj_embed, map_embed],
            dim=-1
        )

        fusion = self.fusion_proj(fusion)

        # residual stabilisation (critical)
        fusion = fusion + ego_embed

        # --------------------------------------------------
        # 5. Classification
        # --------------------------------------------------

        logits = self.classifier(fusion)

        return logits

    # =====================================================
    # TRAINING
    # =====================================================

    def training_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        loss = self.loss_fn(logits, targets)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    # =====================================================
    # VALIDATION
    # =====================================================

    def validation_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        preds = torch.argmax(logits, dim=1)

        self.val_precision.update(preds, targets)

    def on_validation_epoch_end(self):

        precision = self.val_precision.compute()

        self.log("val_precision", precision, prog_bar=True)

        if self.global_rank == 0:
            print(f"\nPrecision (macro): {precision:.4f}")

        self.val_precision.reset()

    # =====================================================
    # OPTIMIZER
    # =====================================================

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
