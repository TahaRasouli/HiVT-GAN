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
            out = self.backbone(batch)
            if self.global_step == 0:
                print("ego_idx min/max:", int(ego_idx.min().cpu()), int(ego_idx.max().cpu()))
                print("nodes_per_graph min/max:", int(nodes_per_graph.min().cpu()), int(nodes_per_graph.max().cpu()))

                print("backbone out shape:", tuple(out.shape))
                if hasattr(batch, "ptr"):
                    print("ptr[-1] total_nodes:", int(batch.ptr[-1]))
                print("num_graphs:", int(batch.num_graphs))


        num_graphs = int(batch.num_graphs)
        assert hasattr(batch, "ego_index"), "Batch missing ego_index"
        assert batch.ego_index.numel() == num_graphs, (
            f"ego_index must have length num_graphs={num_graphs}, got {batch.ego_index.numel()}"
        )

        # Normalize to tensor
        global_embed = out
        if not torch.is_tensor(global_embed):
            raise RuntimeError(f"Backbone returned non-tensor type: {type(global_embed)}")

        # -------- Case A: per-graph embeddings already --------
        # [num_graphs, D]
        if global_embed.dim() == 2 and global_embed.size(0) == num_graphs:
            ego_embeds = global_embed  # already one embedding per graph

        # [1, num_graphs, D]
        elif global_embed.dim() == 3 and global_embed.size(0) == 1:
            assert hasattr(batch, "ptr"), "Batch missing ptr (required for flattened node offsets)"

            ptr = batch.ptr.long()
            ego_idx = batch.ego_index.to(global_embed.device).long()  # [B]
            nodes_per_graph = (ptr[1:] - ptr[:-1]).to(ego_idx.device)  # [B]

            total_nodes = int(global_embed.size(1))

            # Case C1: ego_index is LOCAL within each graph
            if torch.all((ego_idx >= 0) & (ego_idx < nodes_per_graph)):
                ego_global = (ptr[:-1].to(ego_idx.device) + ego_idx)

            # Case C2: ego_index is already GLOBAL
            elif torch.all((ego_idx >= 0) & (ego_idx < total_nodes)):
                ego_global = ego_idx

            else:
                raise RuntimeError(
                    f"ego_index is neither local nor global valid. "
                    f"ego_idx min={int(ego_idx.min().cpu())}, max={int(ego_idx.max().cpu())}, "
                    f"nodes_per_graph min={int(nodes_per_graph.min().cpu())}, max={int(nodes_per_graph.max().cpu())}, "
                    f"total_nodes={total_nodes}"
                )

            # Final bounds check (prevents CUDA abort)
            max_idx = int(ego_global.max().detach().cpu())
            min_idx = int(ego_global.min().detach().cpu())
            if min_idx < 0 or max_idx >= total_nodes:
                raise RuntimeError(
                    f"Computed ego_global out of bounds. min={min_idx}, max={max_idx}, total_nodes={total_nodes}"
                )

            ego_embeds = global_embed[0, ego_global, :]  # [B, D]

        # -------- Case B: graph-batched node embeddings --------
        # [num_graphs, N, D]  (N nodes-per-graph, padded or fixed)
        elif global_embed.dim() == 3 and global_embed.size(0) == 1:
            assert hasattr(batch, "ptr"), "Batch missing ptr"

            ptr = batch.ptr.long()
            ego_idx = batch.ego_index.to(global_embed.device).long()  # <-- define first
            nodes_per_graph = (ptr[1:] - ptr[:-1]).to(ego_idx.device)

            # DEBUG (safe now)
            if self.global_step == 0:
                print("ego_idx shape:", ego_idx.shape)
                print("ego_idx min/max:", int(ego_idx.min().cpu()), int(ego_idx.max().cpu()))
                print("nodes_per_graph min/max:",
                    int(nodes_per_graph.min().cpu()),
                    int(nodes_per_graph.max().cpu()))
                print("ptr[-1] total_nodes:", int(ptr[-1].cpu()))

            total_nodes = global_embed.size(1)

            # Case 1: ego_index is LOCAL
            if torch.all((ego_idx >= 0) & (ego_idx < nodes_per_graph)):
                ego_global = ptr[:-1].to(ego_idx.device) + ego_idx

            # Case 2: ego_index is already GLOBAL
            elif torch.all((ego_idx >= 0) & (ego_idx < total_nodes)):
                ego_global = ego_idx

            else:
                raise RuntimeError(
                    f"Invalid ego_index values. "
                    f"ego_idx min={int(ego_idx.min())}, max={int(ego_idx.max())}, "
                    f"nodes_per_graph max={int(nodes_per_graph.max())}, "
                    f"total_nodes={total_nodes}"
                )

            # Final safety check
            if ego_global.min() < 0 or ego_global.max() >= total_nodes:
                raise RuntimeError(
                    f"Computed ego_global out of bounds: "
                    f"min={int(ego_global.min())}, "
                    f"max={int(ego_global.max())}, "
                    f"total_nodes={total_nodes}"
                )

            ego_embeds = global_embed[0, ego_global, :]


        # -------- Case C: single-batch node embeddings --------
        # [1, total_nodes, D] or [total_nodes, D]
        elif global_embed.dim() == 3 and global_embed.size(0) == 1:
            assert hasattr(batch, "ptr"), "Batch missing ptr (required for flattened node offsets)"
            ego_idx = batch.ego_index.long()

            # ptr gives node offsets into flattened node dimension
            ptr = batch.ptr.long()  # [num_graphs+1]
            ego_global = (ptr[:-1] + ego_idx).to(global_embed.device)  # [num_graphs]

            total_nodes = int(global_embed.size(1))
            max_idx = int(ego_global.max().detach().cpu())
            min_idx = int(ego_global.min().detach().cpu())
            if min_idx < 0 or max_idx >= total_nodes:
                raise RuntimeError(
                    f"Computed ego_global out of bounds for flattened node dim. "
                    f"min={min_idx}, max={max_idx}, total_nodes={total_nodes}, "
                    f"ptr[-1]={int(ptr[-1])}, num_graphs={num_graphs}, "
                    f"global_embed.shape={tuple(global_embed.shape)}"
                )

            ego_embeds = global_embed[0, ego_global, :]  # [num_graphs, D]

        elif global_embed.dim() == 2:
            # [total_nodes, D]
            assert hasattr(batch, "ptr"), "Batch missing ptr (required for flattened node offsets)"
            ego_idx = batch.ego_index.long()
            ptr = batch.ptr.long()
            ego_global = (ptr[:-1] + ego_idx).to(global_embed.device)

            total_nodes = int(global_embed.size(0))
            max_idx = int(ego_global.max().detach().cpu())
            min_idx = int(ego_global.min().detach().cpu())
            if min_idx < 0 or max_idx >= total_nodes:
                raise RuntimeError(
                    f"Computed ego_global out of bounds for flattened node dim. "
                    f"min={min_idx}, max={max_idx}, total_nodes={total_nodes}, "
                    f"ptr[-1]={int(ptr[-1])}, num_graphs={num_graphs}, "
                    f"global_embed.shape={tuple(global_embed.shape)}"
                )

            ego_embeds = global_embed[ego_global, :]  # [num_graphs, D]

        else:
            raise RuntimeError(
                f"Unsupported backbone output shape {tuple(global_embed.shape)} for num_graphs={num_graphs}"
            )

        logits = self.head(ego_embeds)  # [num_graphs, num_classes]
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
