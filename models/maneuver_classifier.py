import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import MulticlassF1Score


# =====================================================
# Trajectory Encoder
# =====================================================

class TrajectoryEncoder(nn.Module):

    def __init__(self, future_steps=30, embed_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(future_steps * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, traj):
        # traj: [B,T,2]
        B = traj.size(0)
        traj_flat = traj.reshape(B, -1)
        return self.net(traj_flat)


# =====================================================
# Ego-Centric Map Encoder (token output)
# =====================================================

class EgoCentricMapEncoder(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, batch, ego_embed):

        lane_vectors = batch.lane_vectors  # [total_lanes,2]

        batch_index = batch.batch

        ego_indices = torch.cat([
            torch.tensor([0], device=batch_index.device),
            torch.where(batch_index[1:] != batch_index[:-1])[0] + 1
        ])

        ego_positions = batch.positions[ego_indices, -1]

        lane_counts = batch.ptr_lane if hasattr(batch, "ptr_lane") else None

        # fallback (assume single scene batch)
        if lane_counts is None:
            lane_feat = self.mlp(lane_vectors)
            return lane_feat.unsqueeze(0)

        lane_feat = self.mlp(lane_vectors)

        B = ego_positions.size(0)

        outputs = []

        start = 0
        for i in range(B):

            end = lane_counts[i+1]

            lanes_i = lane_feat[start:end]

            outputs.append(lanes_i)

            start = end

        return torch.stack(outputs)


# =====================================================
# Cross Attention Fusion
# =====================================================

class ManeuverFusion(nn.Module):

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

    def forward(self, traj_embed, lane_tokens):

        query = traj_embed.unsqueeze(1)

        fused, _ = self.cross_attn(
            query,
            lane_tokens,
            lane_tokens
        )

        return fused.squeeze(1)


# =====================================================
# Maneuver Classifier
# =====================================================

class ManeuverClassifier(pl.LightningModule):

    def __init__(
        self,
        frozen_backbone,
        num_classes=6,
        lr=5e-4,
        class_weights=None
    ):
        super().__init__()

        self.encoder = frozen_backbone

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        embed_dim = self.encoder.hparams.embed_dim

        self.traj_encoder = TrajectoryEncoder(embed_dim=embed_dim)

        self.map_encoder = EgoCentricMapEncoder(embed_dim=embed_dim)

        self.fusion = ManeuverFusion(embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.val_f1_macro = MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        )

        self.val_f1_per_class = MulticlassF1Score(
            num_classes=num_classes,
            average=None
        )

        self.lr = lr

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, batch):

        node_features = self.encoder(batch)   # [1,N,D]
        node_features = node_features.squeeze(0)

        batch_index = batch.batch

        ego_indices = batch.ego_index.view(-1)

        traj = batch.y[ego_indices]   # [B,T,2]

        traj_embed = self.traj_encoder(traj)

        # -------------------------------------------------
        # Ego-centric map encoding
        # -------------------------------------------------

        lane_tokens = self.map_encoder(batch, ego_embed)

        # -------------------------------------------------
        # Cross attention fusion
        # -------------------------------------------------

        fused_motion = self.fusion(traj_embed, lane_tokens)

        fusion = torch.cat([ego_embed, fused_motion], dim=-1)

        logits = self.classifier(fusion)

        return logits

    # =====================================================
    # Training
    # =====================================================

    def training_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        loss = self.loss_fn(logits, targets)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    # =====================================================
    # Validation
    # =====================================================

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
                print(f"{i}: {f.item():.4f}")

        self.val_f1_macro.reset()
        self.val_f1_per_class.reset()

    # =====================================================
    # Optimizer
    # =====================================================

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
