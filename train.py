from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch.multiprocessing as mp
import torch

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.cvae import CVAE
from models.hivt import HiVT

# speed boost on Nvidia-A6000
torch.set_float32_matmul_precision('medium')
mp.set_start_method('spawn', force=True)

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

    # 1. Initialize Dataset & Calculate Weights
    # This will trigger the filtering progress bar we added to nuscenes_dataset.py
    train_dataset = NuScenesHiVTDataset(root=args.root, split='train')
    class_weights = calculate_6class_weights(train_dataset)
    
    # 2. Setup DataModule
    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=8
    )

    # 3. Load Frozen Backbone & Model
    print(f"[Info] Loading Frozen Backbone: {args.ckpt_path}")
    backbone = CVAE.load_from_checkpoint(args.ckpt_path, strict=False)
    
    model = ManeuverClassifier(
        frozen_backbone=backbone,
        num_classes=6,
        lr=args.lr,
        class_weights=class_weights
    )

    # 4. Define Callbacks (Fixes the NameError)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_macro",
        mode="max",
        filename="maneuver-classifier-{epoch:02d}-{val_f1_macro:.2f}",
        save_top_k=2
    )
    
    early_stop = EarlyStopping(
        monitor="val_f1_macro", 
        patience=10, 
        mode="max"
    )

    # 5. Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="auto",
        precision="16-mixed",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10
    )

    print("[Info] Starting Maneuver Classification Training...")
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()