import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from collections import Counter

from datasets.nuscenes_dataset import NuScenesHiVTDataset
from datasets.nuscenes_datamodule import NuScenesHiVTDataModule
from models.trajectory_generator import CVAE # Or HiVT
from models.maneuver_classifier import ManeuverClassifier

def calculate_class_weights(dataset):
    print("[Info] Calculating inverse-frequency weights...")
    counts = Counter()
    for i in range(len(dataset)):
        # Quick access to label without full sanitization if possible
        data = dataset.get(i)
        counts[int(data.maneuver_id.item())] += 1
    
    total = sum(counts.values())
    weights = torch.zeros(7)
    for cls in range(7):
        # Inverse frequency: total / (num_classes * count_per_class)
        weights[cls] = total / (7 * counts[cls]) if counts[cls] > 0 else 1.0
    return weights

def main():
    pl.seed_everything(2024)
    parser = ArgumentParser()

    # Paths
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--devices", type=int, default=1)
    
    args = parser.parse_args()

    # 1. Load Data with caps
    train_dataset = NuScenesHiVTDataset(root=args.root, split='train')
    print(f"Dataset initialized with {len(train_dataset)} capped samples.")

    # 2. Re-calculate weights for 6 classes
    # (Use the calculate_class_weights logic but ensure it skips ID 3)
    class_weights = calculate_6class_weights(train_dataset)

    # 3. Initialize Model
    backbone = CVAE.load_from_checkpoint(args.ckpt_path)
    model = ManeuverClassifier(
        frozen_backbone=backbone,
        num_classes=6,
        lr=args.lr,
        class_weights=class_weights
    )

    # 4. Run Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[checkpoint_callback, early_stop]
    )
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()