import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassPrecision


# ============================================================
# Trajectory Encoder
# ============================================================

class TrajectoryEncoder(nn.Module):

    def __init__(self, future_steps=30, embed_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(future_steps * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, traj):

        # traj: [B, T, 2]

        B = traj.shape[0]

        traj_flat = traj.reshape(B, -1)   # [B, T*2]

        return self.net(traj_flat)        # [B, D]


# ============================================================
# Ego-centric Map Encoder
# ============================================================

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

        # lane_vectors: [L_total,2]
        lane_vecs = batch.lane_vectors

        lane_embed = self.lane_mlp(lane_vecs)   # [L_total, D]

        # We assume global lane pool (consistent with your dataset)
        lanes = lane_embed.unsqueeze(0)  # [1,L,D]

        outputs = []

        B = ego_embed.shape[0]

        for i in range(B):

            ego = ego_embed[i].unsqueeze(0).unsqueeze(0)  # [1,1,D]

            attn_out, _ = self.attn(ego, lanes, lanes)

            outputs.append(attn_out.squeeze(0))

        return torch.cat(outputs, dim=0)   # [B,D]


# ============================================================
# Maneuver Classifier
# ============================================================

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

        # -----------------------------------------
        # Backbone
        # -----------------------------------------
        self.encoder = frozen_backbone

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        embed_dim = self.encoder.hparams.embed_dim

        # -----------------------------------------
        # Modules
        # -----------------------------------------
        self.traj_encoder = TrajectoryEncoder(
            future_steps=future_steps,
            embed_dim=embed_dim
        )

        self.map_encoder = EgoCentricMapEncoder(
            embed_dim=embed_dim
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),

            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )


        # -----------------------------------------
        # Loss
        # -----------------------------------------
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # -----------------------------------------
        # Metrics
        # -----------------------------------------
        self.val_precision_macro = MulticlassPrecision(
            num_classes=num_classes,
            average="macro"
        )

        self.val_precision_per_class = MulticlassPrecision(
            num_classes=num_classes,
            average=None
        )

        self.lr = lr

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, batch):

        # -----------------------------------------
        # 1. Scene encoding (HiVT backbone)
        # -----------------------------------------

        node_features = self.encoder(batch)  # [1,N_total,D]
        node_features = node_features.squeeze(0)  # [N_total,D]

        batch_index = batch.batch

        # Ego extraction
        ego_indices = torch.cat([
            torch.tensor([0], device=batch_index.device),
            torch.where(batch_index[1:] != batch_index[:-1])[0] + 1
        ])

        ego_embed = node_features[ego_indices]  # [B,D]

        # -----------------------------------------
        # 2. Trajectory encoder (GROUND TRUTH)
        # -----------------------------------------

        # data.y = [num_nodes, T, 2]
        traj = batch.y[ego_indices]   # select ego GT trajectory

        traj_embed = self.traj_encoder(traj)  # [B,D]

        # -----------------------------------------
        # 3. Map encoder (ego-centric)
        # -----------------------------------------

        map_embed = self.map_encoder(batch, ego_embed)  # [B,D]

        # -----------------------------------------
        # 4. Fusion
        # -----------------------------------------

        fusion = torch.cat(
            [ego_embed, traj_embed, map_embed],
            dim=-1
        )  # [B,3D]

        logits = self.classifier(fusion) / 1.5 # preventing the overconfident predictions!

        return logits

    # ============================================================
    # Training
    # ============================================================

    def training_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        loss = self.loss_fn(logits, targets)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    # ============================================================
    # Validation
    # ============================================================

    def validation_step(self, batch, batch_idx):

        logits = self(batch)

        targets = batch.maneuver_id.view(-1)

        preds = torch.argmax(logits, dim=1)

        self.val_precision_macro.update(preds, targets)
        self.val_precision_per_class.update(preds, targets)

    # ============================================================
    # Epoch End
    # ============================================================

    def on_validation_epoch_end(self):

        precision_macro = self.val_precision_macro.compute()
        precision_per_class = self.val_precision_per_class.compute()

        self.log("val_precision", precision_macro, prog_bar=True)

        if self.global_rank == 0:
            print("\n==== Per-class Precision ====")
            for i, p in enumerate(precision_per_class):
                print(f"Class {i}: {p.item():.4f}")

        self.val_precision_macro.reset()
        self.val_precision_per_class.reset()

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):

        return torch.optim.AdamW(self.parameters(), lr=self.lr)
