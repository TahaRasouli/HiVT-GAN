from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch


from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT

# gives a significant speed boost on Nvidia-A6000
torch.set_float32_matmul_precision('medium') # or 'high'


def main():
    pl.seed_everything(2022)

    parser = ArgumentParser()

    # -----------------------------
    # Data arguments
    # -----------------------------
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing train_processed/ and val_processed/")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--ckpt_path", type=str, default=None)


    # -----------------------------
    # Training arguments
    # -----------------------------
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_minFDE",
        choices=["val_minADE", "val_minFDE", "val_minMR"],
    )
    parser.add_argument("--save_top_k", type=int, default=5)


    # -----------------------------
    # HiVT model arguments
    # -----------------------------
    parser = HiVT.add_model_specific_args(parser)

    args = parser.parse_args()

    # -----------------------------
    # Callbacks
    # -----------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode="min",
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    # train.py
    strategy = DDPStrategy(find_unused_parameters=True) # Always True for GANs

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        # Add gradient clipping here for GAN stability
        gradient_clip_val=0.5,
        max_epochs=args.max_epochs,
    )


    # -----------------------------
    # Model
    # -----------------------------
    model = HiVT(**vars(args))

    if args.ckpt_path and args.use_gan:
        # Manual weight loading to bridge the non-GAN -> GAN gap
        print(f"Loading pre-trained HiVT weights from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=model.device)
        # strict=False is KEY: it loads HiVT weights and ignores missing Critic weights
        model.load_state_dict(ckpt['state_dict'], strict=False)
        # We clear ckpt_path so trainer.fit doesn't try to restore optimizers
        args.ckpt_path = None

    # -----------------------------
    # DataModule
    # -----------------------------
    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    main()
