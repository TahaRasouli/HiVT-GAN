import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.cvae_gan import CVAE_GAN
from torch_geometric.nn import global_max_pool

class CaptionFinetuner(pl.LightningModule):
    def __init__(self, pretrained_ckpt):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load Backbone
        print(f"Loading backbone from {pretrained_ckpt}...")
        self.model = CVAE_GAN.load_from_checkpoint(pretrained_ckpt, strict=False)

        # 2. Define the Classifier (The Head)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 7) 
        )
        
        # 3. Freeze Backbone
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Note: We do NOT need to manually unfreeze self.classifier.
        # Since it was just created in __init__, requires_grad is True by default.

        # 4. Weighted Loss
        weights = torch.tensor([
            1.0,   # 0: Straight
            15.0,  # 1: Left
            15.0,  # 2: Right
            50.0,  # 3: U-Turn
            25.0,  # 4: Lane L
            25.0,  # 5: Lane R
            5.0    # 6: Stop
        ])
        self.register_buffer("class_weights", weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.validation_step_outputs = []

    def forward(self, data):
        # 1. Get Embeddings
        local_embed = self.model.local_encoder(data)
        global_embed = self.model.global_interactor(data, local_embed)
        
        # --- DEBUG PRINT (Optional: Remove after fixing) ---
        # print(f"Global Embed Shape: {global_embed.shape}")
        # print(f"Batch Vector Shape: {data.batch.shape}")
        # ---------------------------------------------------

        # 2. SELECT EGO AGENT (Fix for Batch Size 1 Error)
        # Instead of pooling, we grab the 0-th node of every graph in the batch.
        # data.ptr contains the start index of each graph in the batch.
        # data.ptr[:-1] gives us the indices [0, num_nodes_1, num_nodes_1+num_nodes_2, ...]
        if hasattr(data, 'ptr'):
            ego_indices = data.ptr[:-1]
            graph_embed = global_embed[ego_indices]
        else:
            # Fallback for some PyG versions or if ptr is missing (unlikely)
            # This replicates "Select index 0 where batch changes"
            # But relying on ptr is safer for HiVT
            graph_embed = global_max_pool(global_embed, data.batch)

        # 3. CLASSIFICATION
        # Now graph_embed is GUARANTEED to be [Batch_Size, Hidden_Dim]
        logits = self.classifier(graph_embed) 
        
        return logits

    def training_step(self, data, batch_idx):
        logits = self(data)
        target = data.maneuver_id.squeeze()
        
        loss = self.ce_loss(logits, target)
        
        acc = (torch.argmax(logits, dim=1) == target).float().mean()
        
        # Use data.num_graphs for correct batch size logging in PyG
        self.log("train_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        self.log("train_acc", acc, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        target = data.maneuver_id.squeeze()
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
        # --- FIX: Optimize self.classifier, NOT self.model.decoder... ---
        return torch.optim.Adam(self.classifier.parameters(), lr=1e-3)