import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score


# =========================================================
# 1. TRAJECTORY ENCODER
# =========================================================

class TrajectoryEncoder(nn.Module):

    def __init__(self, future_steps, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(future_steps * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, traj):
        # traj: [B,T,2]
        traj_flat = traj.reshape(traj.shape[0], -1)
        return self.net(traj_flat)


# =========================================================
# 2. ADVANCED EGO-CENTRIC MAP ENCODER
# =========================================================

class EgoCentricMapEncoder(nn.Module):

    def __init__(self, embed_dim=128, radius=40.0, num_heads=4):
        super().__init__()

        self.radius = radius

        self.lane_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

    def forward(self, batch, ego_embed):

        device = ego_embed.device
        lane_vectors = batch.lane_vectors.to(device)

        B = ego_embed.size(0)
        batch_index = batch.batch

        # ego indices
        ego_indices = torch.cat([
            torch.tensor([0], device=device),
            torch.where(batch_index[1:] != batch_index[:-1])[0] + 1
        ])

        ego_positions = batch.positions[ego_indices, -1]

        outputs = []

        for b in range(B):

            ego_pos = ego_positions[b]

            rel_lane = lane_vectors - ego_pos
            dist = torch.norm(rel_lane, dim=-1)

            mask = dist < self.radius
            local_lanes = rel_lane[mask]

            if local_lanes.shape[0] == 0:
                outputs.append(torch.zeros_like(ego_embed[b]))
                continue

            lane_feat = self.lane_embed(local_lanes).unsqueeze(0)

            ego_query = ego_embed[b].unsqueeze(0).unsqueeze(0)

            attn_out, _ = self.cross_attn(
                ego_query,
                lane_feat,
                lane_feat
            )

            outputs.append(attn_out.squeeze(0).squeeze(0))

        return torch.stack(outputs)


# =========================================================
# 3. MANEUVER CLASSIFIER
# =========================================================

class ManeuverClassifier(pl.LightningModule):

    def __init__(
        self,
        frozen_backbone,
        num_classes=6,
        lr=5e-4,
        class_weights=None,
        future_steps=30,
        id_to_class=None
    ):
        super().__init__()

        # ------------------------------------------------
        # Backbone
        # ------------------------------------------------

        self.encoder = frozen_backbone

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        embed_dim = self.encoder.hparams.embed_dim

        # ------------------------------------------------
        # Modules
        # ------------------------------------------------

        self.traj_encoder = TrajectoryEncoder(
            future_steps,
            embed_dim
        )

        self.map_encoder = EgoCentricMapEncoder(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.lr = lr
        self.id_to_class = id_to_class

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

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(self, batch):

        # Scene encoding (HiVT backbone)
        node_features = self.encoder(batch).squeeze(0)

        batch_index = batch.batch

        ego_indices = torch.cat([
            torch.tensor([0], device=batch_index.device),
            torch.where(batch_index[1:] != batch_index[:-1])[0] + 1
        ])

        ego_embed = node_features[ego_indices]

        # Trajectory encoding
        traj = batch.y[ego_indices]
        traj_embed = self.traj_encoder(traj)

        # Map encoding
        map_embed = self.map_encoder(batch, ego_embed)

        # Fusion
        fusion = torch.cat([
            ego_embed,
            map_embed,
            traj_embed
        ], dim=-1)

        logits = self.classifier(fusion)

        return logits

    # =========================================================
    # TRAINING
    # =========================================================

    def training_step(self, batch, batch_idx):

        logits = self(batch)
        targets = batch.maneuver_id.view(-1)

        loss = self.loss_fn(logits, targets)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    # =========================================================
    # VALIDATION
    # =========================================================

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

                name = (
                    self.id_to_class[i]
                    if self.id_to_class is not None
                    else str(i)
                )

                print(f"{name}: {f.item():.4f}")

        self.val_f1_macro.reset()
        self.val_f1_per_class.reset()

    def configure_optimizers(self):

        return torch.optim.AdamW(self.parameters(), lr=self.lr)
