import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from collections import Counter

from datasets.nuscenes_dataset import NuScenesHiVTDataset
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.cvae import CVAE
from models.maneuver_classifier import ManeuverClassifier

torch.set_float32_matmul_precision('high')

# -------------------------------------------------
# class weights (UNCHANGED)
# -------------------------------------------------
def calculate_6class_weights(dataset):
    print("[Info] Calculating weights for 6-class setup...")
    counts = Counter()
    mapping = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5}

    for i in range(len(dataset)):
        m_id = int(dataset.get(i).maneuver_id.item())
        if m_id in mapping:
            counts[mapping[m_id]] += 1

    total = sum(counts.values())
    weights = torch.zeros(6)

    for idx in range(6):
        raw_weight = total / (6 * counts[idx]) if counts[idx] > 0 else 1.0
        weights[idx] = min(raw_weight, 20.0)

    print(f"[Info] Remapped & Capped Weights: {weights}")
    return weights


# -------------------------------------------------
# main
# -------------------------------------------------
def main():

    pl.seed_everything(2024)

    parser = ArgumentParser()

    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")

    args = parser.parse_args()

    # -------------------------
    # Dataset (UNCHANGED)
    # -------------------------
    train_dataset = NuScenesHiVTDataset(
        root=args.root,
        split='train'
    )

    class_weights = calculate_6class_weights(train_dataset)

    # -------------------------
    # DataModule (UNCHANGED)
    # -------------------------
    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        train_transform=None,
        val_transform=None,
        max_train_samples=None,
        max_val_samples=None
    )

    # -------------------------
    # Backbone load (UNCHANGED)
    # -------------------------
    print(f"[Info] Loading Backbone: {args.ckpt_path}")
    backbone = CVAE.load_from_checkpoint(
        args.ckpt_path,
        strict=False
    )

    # -------------------------
    # Model (MINIMAL CHANGE HERE)
    # -------------------------
    model = ManeuverClassifier(
        encoder=backbone,              # ← name change
        embed_dim=128,                 # ← match your backbone output dim if different
        num_classes=6,
        lr=args.lr,
        loss_weights=class_weights     # ← name change
    )

    # -------------------------
    # Callbacks (UNCHANGED)
    # -------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_macro",
        mode="max",
        filename="maneuver-{epoch:02d}-{val_f1_macro:.2f}",
        save_top_k=2
    )

    early_stop = EarlyStopping(
        monitor="val_f1_macro",
        patience=10,
        mode="max"
    )

    # -------------------------
    # Trainer (UNCHANGED)
    # -------------------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        gradient_clip_val=0.5,
        precision="32",
    )

    # -------------------------
    # Fit
    # -------------------------
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
