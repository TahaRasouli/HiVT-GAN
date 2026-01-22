import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cvae_gan import CVAE_GAN
from torch_geometric.nn import global_max_pool

class CaptionFinetuner(pl.LightningModule):
    def __init__(self, pretrained_ckpt):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load Backbone (strict=False to ignore original decoder heads if needed)
        print(f"Loading backbone from {pretrained_ckpt}...")
        self.model = CVAE_GAN.load_from_checkpoint(pretrained_ckpt, strict=False)

        # 2. Define the Classifier Head
        # HiVT hidden_dim is usually 128.
        # Output is 7 classes (based on your MANEUVER_MAP).
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
            
        # 4. Class Weights (To handle imbalanced maneuvers like U-Turns)
        weights = torch.tensor([
            1.0,   # 0: Straight
            10.0,  # 1: Left
            10.0,  # 2: Right
            50.0,  # 3: U-Turn
            20.0,  # 4: Lane L
            20.0,  # 5: Lane R
            5.0    # 6: Stop
        ])
        self.register_buffer("class_weights", weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.validation_step_outputs = []

    def forward(self, data):
        # 1. Get Local Embeddings
        local_embed = self.model.local_encoder(data)
        
        # 2. Get Global Interactions
        global_embed = self.model.global_interactor(data, local_embed)
        
        # 3. SELECT EGO AGENT (Critical Fix for Batch Size Error)
        # HiVT places the Ego agent at index 0 of every graph.
        # data.ptr contains the start index of every graph in the batch.
        # We use this to pick the specific embedding for the ego vehicle.
        if hasattr(data, 'ptr'):
            ego_indices = data.ptr[:-1]
            graph_embed = global_embed[ego_indices]
        else:
            # Fallback for old PyG versions
            graph_embed = global_max_pool(global_embed, data.batch)

        # 4. Classification
        logits = self.classifier(graph_embed) 
        
        return logits

    def training_step(self, data, batch_idx):
        logits = self(data)
        target = data.maneuver_id.squeeze()
        
        loss = self.ce_loss(logits, target)
        
        acc = (torch.argmax(logits, dim=1) == target).float().mean()
        
        # Logging
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
        # Only optimize the classifier head!
        return torch.optim.Adam(self.classifier.parameters(), lr=1e-3)