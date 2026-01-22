import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score


class ManeuverClassifier(pl.LightningModule):
    """
    Ego-centric maneuver classifier on top of a frozen HiVT / CVAE backbone.

    Backbone output: [B, N, 128]
    Ego pooling via batch.ego_index → [B, 128]
    """

    def __init__(
        self,
        frozen_backbone,
        num_classes: int = 7,
        learning_rate: float = 1e-3,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["frozen_backbone"])

        # ----------------------------------------------------------
        # 1. FROZEN BACKBONE
        # ----------------------------------------------------------
        self.backbone = frozen_backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ----------------------------------------------------------
        # 2. CLASSIFICATION HEAD
        # ----------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

        # ----------------------------------------------------------
        # 3. LOSS + METRICS
        # ----------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )

        self.class_names = [
            "Straight",
            "Left Turn",
            "Right Turn",
            "U-Turn",
            "LC Left",
            "LC Right",
            "Stationary",
        ]

    # --------------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------------
    def forward(self, batch):
        self.backbone.eval()
        with torch.no_grad():
            out = self.backbone(batch)  # expected [1, total_nodes, D]

        # --------------------------------------------------
        # Basic checks
        # --------------------------------------------------
        if not torch.is_tensor(out):
            raise RuntimeError(f"Backbone returned non-tensor: {type(out)}")

        if out.dim() != 3 or out.size(0) != 1:
            raise RuntimeError(
                f"Expected backbone output [1, total_nodes, D], got {tuple(out.shape)}"
            )

        assert hasattr(batch, "ptr"), "Batch missing ptr"
        assert hasattr(batch, "ego_index"), "Batch missing ego_index"

        ptr = batch.ptr.long()                    # [num_graphs + 1]
        ego_idx = batch.ego_index.long()          # [num_graphs]
        num_graphs = int(batch.num_graphs)
        total_nodes = int(out.size(1))

        if ego_idx.numel() != num_graphs:
            raise RuntimeError(
                f"ego_index length mismatch: expected {num_graphs}, got {ego_idx.numel()}"
            )

        # --------------------------------------------------
        # Compute global ego indices (LOCAL → GLOBAL)
        # --------------------------------------------------
        nodes_per_graph = ptr[1:] - ptr[:-1]

        if not torch.all((ego_idx >= 0) & (ego_idx < nodes_per_graph)):
            raise RuntimeError(
                f"Invalid ego_index (must be local per graph). "
                f"ego_idx min={int(ego_idx.min())}, max={int(ego_idx.max())}, "
                f"nodes_per_graph min={int(nodes_per_graph.min())}, "
                f"max={int(nodes_per_graph.max())}"
            )

        ego_global = ptr[:-1] + ego_idx  # [num_graphs]

        # --------------------------------------------------
        # Final bounds check (CUDA-safe)
        # --------------------------------------------------
        if ego_global.min() < 0 or ego_global.max() >= total_nodes:
            raise RuntimeError(
                f"Computed ego_global out of bounds. "
                f"min={int(ego_global.min())}, max={int(ego_global.max())}, "
                f"total_nodes={total_nodes}"
            )

        # --------------------------------------------------
        # DEBUG (only once)
        # --------------------------------------------------
        if self.global_step == 0:
            print("backbone out shape:", tuple(out.shape))
            print("num_graphs:", num_graphs)
            print("total_nodes:", total_nodes)
            print("ego_idx min/max:",
                int(ego_idx.min()), int(ego_idx.max()))
            print("ego_global min/max:",
                int(ego_global.min()), int(ego_global.max()))

        # --------------------------------------------------
        # Extract ego embeddings
        # --------------------------------------------------
        ego_embeds = out[0, ego_global, :]  # [num_graphs, D]

        logits = self.head(ego_embeds)      # [num_graphs, num_classes]
        return logits



    # --------------------------------------------------------------
    # TRAINING STEP
    # --------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        targets = batch.maneuver_id.view(-1).long()
        logits = self(batch)

        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, targets)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )

        return loss

    # --------------------------------------------------------------
    # VALIDATION STEP
    # --------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            print("ego_index shape:", batch.ego_index.shape)

        logits = self(batch)
        targets = batch.maneuver_id.view(-1)

        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, targets)
        self.val_f1_per_class.update(preds, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )

        return loss

    # --------------------------------------------------------------
    # VALIDATION EPOCH END
    # --------------------------------------------------------------
    def on_validation_epoch_end(self):
        self.val_acc.reset()
        self.val_f1_per_class.reset()

    # --------------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.head.parameters(),
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
