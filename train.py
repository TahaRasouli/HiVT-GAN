from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT


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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,       
        strategy=pl.strategies.DDPStrategy(
        find_unused_parameters=False
        ),
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
    )


    # -----------------------------
    # Model
    # -----------------------------
    model = HiVT(**vars(args))

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
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
