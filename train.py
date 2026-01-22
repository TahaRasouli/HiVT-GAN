import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.caption import CaptionFinetuner

def main():
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to processed dataset')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to pre-trained HiVT checkpoint')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # 1. Initialize Tokenizer (Required by your Dataset class)
    print("Initializing BERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # 2. DataModule
    print("Setting up DataModule...")
    # Your dataset expects split_file to be inside root or passed explicitly. 
    # Assuming 'split_datas.json' is in the root folder based on your previous logs.
    split_path = os.path.join(args.root, "split_datas.json")
    
    datamodule = NuScenesHiVTDataModule(
        root=args.root,
        split_file=split_path,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        tokenizer=tokenizer, # Pass tokenizer here
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    # 3. Model
    print("Initializing Model...")
    model = CaptionFinetuner(pretrained_ckpt=args.ckpt_path)

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='caption-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_acc',
        patience=5,
        mode='max'
    )

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.devices,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10,
        strategy='ddp_find_unused_parameters_true' if args.devices > 1 else 'auto'
    )

    # 6. Fit
    print("Starting Training...")
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    main()