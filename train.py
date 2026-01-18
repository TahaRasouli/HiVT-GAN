import torch.utils._pytree
import sys

if not hasattr(torch.utils._pytree, 'register_pytree_node'):
    torch.utils._pytree.register_pytree_node = torch.utils._pytree._register_pytree_node

# Patch Lightning Utilities if needed
try:
    import lightning_utilities
    if not hasattr(lightning_utilities, 'module_available'):
        from lightning_utilities.core.imports import module_available
        lightning_utilities.module_available = module_available
except ImportError:
    pass

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
# Removed unused BLEUScore import
import torch.multiprocessing as mp
import torch
import torch.nn as nn

from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
from models.hivt import HiVT
from models.cvae_gan import CVAE_GAN
from models.hivt_x import HiVTX   
from transformers import AutoTokenizer

# speed boost on Nvidia-A6000
torch.set_float32_matmul_precision('medium')
mp.set_start_method('spawn', force=True)

def main():
    pl.seed_everything(2022)
    parser = ArgumentParser()

    # Data arguments
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--ckpt_path", type=str, default=None)

    # Training arguments
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--monitor", type=str, default="val_minFDE")
    parser.add_argument("--save_top_k", type=int, default=5)
    
    # --- MODEL FLAGS ---
    parser.add_argument("--train_cvae_gan", action="store_true")
    parser.add_argument("--train_contrastive", action="store_true", help="Train HiVT-X Contrastive Model")
    
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")

    # HiVT model specific args
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    # 1. Lower the Learning Rate for fine-tuning
    if args.ckpt_path:
        print(f"Fine-tuning detected. Lowering Learning Rate to 1e-4")
        args.lr = 1e-4 

    # 2. DATA MODULE
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    datamodule = NuScenesHiVTDataModule(
        split_file="balanced_splits.json", 
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        tokenizer=tokenizer # <--- Pass the BERT tokenizer
    )
    datamodule.prepare_data() 
    print(f"--- DataModule Ready with DistilBERT Tokenizer ---")

    # 3. MODEL INITIALIZATION
    actual_fit_path = args.ckpt_path
    
    # CASE A: Contrastive Learning (HiVT-X)
    
    if args.train_contrastive:
        print("--- initializing HiVT-X (Contrastive + DistilBERT) ---")
        # Remove vocab_size arg, HiVTX doesn't need it anymore
        model = HiVTX(cvae_gan_ckpt=args.ckpt_path, **vars(args))
        actual_fit_path = None

    # CASE B: CVAE-GAN
    elif args.train_cvae_gan:
        print("--- initializing CVAE_GAN Model ---")
        args.num_modes = 1 
        model = CVAE_GAN(**vars(args))
        
    # CASE C: Standard HiVT
    else:
        print("--- initializing Standard HiVT Model ---")
        model = HiVT(**vars(args))

    # =====================================================
    # 4. WARM START / TRANSFER LOGIC (For Cases B & C)
    # =====================================================
    # FIX: Changed 'args.train_caption' to 'args.train_contrastive'
    if args.ckpt_path and not args.train_contrastive:
        print(f"--- Loading Weights from: {args.ckpt_path} ---")
        
        if args.freeze_encoder:
            print("(!) Freezing Enabled: Loading weights manually and resetting optimizer.")
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['state_dict'], strict=False)
            actual_fit_path = None 
        else:
            # Smart Transfer Logic
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            state_dict = ckpt['state_dict']
            is_gan_checkpoint = any("critic" in k for k in state_dict.keys())
            
            if not is_gan_checkpoint and args.train_cvae_gan:
                print("Detected CVAE checkpoint -> Transferring to CVAE_GAN.")
                model.load_state_dict(state_dict, strict=False)
                actual_fit_path = None 
            elif not is_gan_checkpoint and not args.train_cvae_gan:
                print("Detected CVAE checkpoint -> Transferring to Standard HiVT.")
                keys_to_remove = [k for k in state_dict.keys() if "decoder" in k or "multihead_proj" in k]
                for k in keys_to_remove: del state_dict[k]
                model.load_state_dict(state_dict, strict=False)
                actual_fit_path = None
            else:
                print("Detected matching checkpoint type. Full resume enabled.")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode="min",
    )

    # Trainer
    strategy = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        precision="16-mixed",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule, ckpt_path=actual_fit_path)

if __name__ == "__main__":
    main()