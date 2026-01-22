import pytorch_lightning as pl
import torch
import torch.nn as nn
from argparse import ArgumentParser
from models.cvae_gan import CVAE_GAN
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule

class CaptionFinetuner(pl.LightningModule):
    def __init__(self, pretrained_ckpt):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load Pre-trained CVAE_GAN
        # strict=False is CRITICAL. It loads the backbone weights but ignores 
        # that the checkpoint is missing the 'decoder.caption_head' weights.
        print(f"Loading backbone from {pretrained_ckpt}...")
        self.model = CVAE_GAN.load_from_checkpoint(pretrained_ckpt, strict=False)
        
        # 2. Freeze the Backbone
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 3. Unfreeze ONLY the Caption Head
        # These are the only weights that will be updated.
        for param in self.model.decoder.caption_head.parameters():
            param.requires_grad = True
            
        self.ce_loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def forward(self, data):
        # Inference Logic: Use the CVAE Prior -> Z -> Decoder
        local_embed = self.model.local_encoder(data)
        global_embed = self.model.global_encoder(data, local_embed)
        _, _, caption_logits = self.model.decoder(global_embed, y_gt=None)
        return caption_logits

    def training_step(self, data, batch_idx):
        # 1. Forward
        # We pass y_gt=None to force the model to use the Prior (inference path)
        # This ensures the classifier learns from the latent space used at runtime.
        logits = self(data)
        
        # 2. Label
        target = data.maneuver_label.squeeze()
        
        # 3. Loss
        loss = self.ce_loss(logits, target)
        
        # 4. Metrics
        acc = (torch.argmax(logits, dim=1) == target).float().mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        self.log("train_acc", acc, prog_bar=True, batch_size=data.num_graphs)
        
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        target = data.maneuver_label.squeeze()
        loss = self.ce_loss(logits, target)
        
        preds = torch.argmax(logits, dim=1)
        correct = (preds == target).float().sum()
        total = torch.tensor(target.numel(), device=self.device)
        
        self.validation_step_outputs.append({
            "loss": loss,
            "correct": correct,
            "total": total
        })
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        total_correct = torch.stack([x["correct"] for x in self.validation_step_outputs]).sum()
        total_samples = torch.stack([x["total"] for x in self.validation_step_outputs]).sum()
        
        val_acc = total_correct / total_samples
        
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        
        if self.global_rank == 0:
            print(f"\n[Epoch {self.current_epoch}] Val Acc: {val_acc*100:.2f}% | Loss: {avg_loss:.4f}")
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Optimize ONLY the new head
        return torch.optim.Adam(self.model.decoder.caption_head.parameters(), lr=1e-3)

def main():
    pl.seed_everything(42)
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to your CVAE checkpoint")
    parser.add_argument("--root", type=str, required=True, help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    datamodule = NuScenesHiVTDataModule(
        root=args.root, 
        split_file="balanced_splits.json", 
        train_batch_size=args.batch_size, 
        val_batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    model = CaptionFinetuner(pretrained_ckpt=args.ckpt_path)
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc", 
                mode="max", 
                filename="fast_caption_head-{epoch:02d}-{val_acc:.2f}"
            )
        ]
    )
    
    print("--- Starting Superfast Caption Head Training ---")
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()