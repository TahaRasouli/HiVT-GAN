import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics.classification import MulticlassF1Score


# =========================================================
# Trajectory Encoder
# =========================================================

class TrajectoryEncoder(nn.Module):
    """
    Encodes ego trajectory [B, T, 2] -> [B, D]
    """

    def __init__(self, future_steps=30, embed_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(future_steps * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, traj):

        # traj shape: [B, T, 2]

        B = traj.size(0)

        traj_flat = traj.reshape(B, -1)   # [B, T*2]

        return self.net(traj_flat)


# =========================================================
# Ego-Centric Map Encoder
# =========================================================

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

        # lane_vectors: [total_lanes, 2]
        lane_vecs = batch.lane_vectors

        lane_embed = self.lane_mlp(lane_vecs)   # [L, D]

        batch_index = batch.batch

        unique_batches = torch.unique(batch_index)

        outputs = []

        start_lane = 0

        for i, b in enumerate(unique_batches):

            num_nodes = (batch_index == b).sum()

            ego = ego_embed[i].unsqueeze(0).unsqueeze(0)   # [1,1,D]

            # NOTE:
            # Simplified — assumes global lanes (works with your current structure)
            lanes = lane_embed.unsqueeze(0)  # [1,L,D]

            attn_out, _ = self.attn(ego, lanes, lanes)

            outputs.append(attn_out.squeeze(0))

        return torch.cat(outputs, dim=0)   # [B,D]


# =========================================================
# Maneuver Classifier
# =========================================================

class ManeuverClassifier(pl.LightningModule):

    def __init__(
        self,
        frozen_backbone,
        num_classes=6,
        lr=5e-4,
        class_weights=None,
        id_to_class=None,
        future_steps=30
    ):
        super().__init__()

        self.encoder = frozen_backbone

        # Freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

        self.lr = lr
        self.future_steps = future_steps

        embed_dim = self.encoder.hparams.embed_dim

        # Components
        self.traj_encoder = TrajectoryEncoder(future_steps, embed_dim)

        self.map_encoder = EgoCentricMapEncoder(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Loss
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        self.val_f1_macro = MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        )

        self.val_f1_per_class = MulticlassF1Score(
            num_classes=num_classes,
            average=None
        )

        self.id_to_class = id_to_class

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, batch):

        # ---------------------------------------------
        # 1. Scene encoding via frozen backbone
        # ---------------------------------------------

        node_features = self.encoder(batch)   # [F, N_total, D]

        node_features = node_features[0]      # use single mode

        # ---------------------------------------------
        # 2. Extract ego embeddings
        # ---------------------------------------------

        ego_indices = batch.ego_index.view(-1)

        ego_embed = node_features[ego_indices]   # [B, D]

        # ---------------------------------------------
        # 3. Ground-truth trajectory encoding
        # ---------------------------------------------

        traj = batch.y[ego_indices]   # [B, T, 2]

        traj_embed = self.traj_encoder(traj)   # [B,D]

        # ---------------------------------------------
        # 4. Ego-centric map encoding
        # ---------------------------------------------

        map_embed = self.map_encoder(batch, ego_embed)   # [B,D]

        # ---------------------------------------------
        # 5. Fusion
        # ---------------------------------------------

        fusion = torch.cat(
            [ego_embed, traj_embed, map_embed],
            dim=-1
        )   # [B,3D]

        logits = self.classifier(fusion)

        return logits

    # =====================================================
    # TRAIN
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
                    if self.id_to_class is not None else i
                )

                print(f"{name}: {f.item():.4f}")

        self.val_f1_macro.reset()
        self.val_f1_per_class.reset()

    # =====================================================
    # OPTIMIZER
    # =====================================================

    def configure_optimizers(self):

        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr
        )
