from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


class GRUDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(GRUDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.repeat(self.num_modes, 1).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
            return torch.cat((loc, scale),
                             dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi  # [F, N, H, 4], [N, F]
        else:
            return loc.view(self.num_modes, -1, self.future_steps, 2), pi  # [F, N, H, 2], [N, F]


class MLPDecoder(nn.Module):
    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(MLPDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        # --- NEW: Mode-specific Latent Code ---
        # This allows the GAN to differentiate the 6 branches
        self.mode_emb = nn.Parameter(torch.randn(num_modes, local_channels) * 0.01)

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
            
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
            
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
                
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Inject mode-specific latent info into the expanded local_embed
        # [F, N, D] = [F, 1, D] + [1, N, D]
        # This keeps the backbone weights valid but differentiates the modes for the GAN
        expanded_local = local_embed.unsqueeze(0).expand(self.num_modes, -1, -1)
        mode_specific_local = expanded_local + self.mode_emb.unsqueeze(1)

        # 2. Probability (pi) uses the modified embedding
        pi = self.pi(torch.cat((mode_specific_local, global_embed), dim=-1)).squeeze(-1).t()
        
        # 3. Aggregated embedding
        out = self.aggr_embed(torch.cat((global_embed, mode_specific_local), dim=-1))
        
        # 4. Trajectory Generation
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)
        
        if self.uncertain:
            # We use the clamp here for stability as discussed before
            scale = F.elu(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = torch.clamp(scale + self.min_scale, min=0.05) 
            return torch.cat((loc, scale), dim=-1), pi
        else:
            return loc, pi


class DiffusionDecoder(nn.Module):
    """
    The Neural Network that predicts the noise.
    Input: Noisy Trajectory + Timestep + HiVT Context
    Output: Predicted Noise
    """
    def __init__(self, embed_dim=128, future_steps=30, out_dim=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        
        # 1. Time Embedding (Sinusoidal like Transformer)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 2. Trajectory Embedding
        # We map the 2D coordinates to a higher dimension
        self.traj_emb = nn.Linear(out_dim, embed_dim)
        
        # 3. Main Denoising Network (ResNet-style MLP)
        # Input size = Traj_Emb + Context_Emb + Time_Emb = 3 * Embed_Dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.Mish(),
            nn.Dropout(0.1),
            
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.Mish(),
            nn.Dropout(0.1),
            
            nn.Linear(embed_dim * 2, out_dim) # Predicts pure noise (epsilon)
        )

    def forward(self, x_noisy, t, context):
        """
        x_noisy: [Batch, FutureSteps, 2]
        t: [Batch]
        context: [Batch, EmbedDim] (From HiVT Global Interactor)
        """
        B, T, _ = x_noisy.shape
        
        # A. Embed Time
        t_emb = self.time_mlp(t.float().unsqueeze(-1)) # [B, Embed]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)   # [B, T, Embed]
        
        # B. Embed Context (Global features from HiVT)
        # Context is originally [B, Embed], expand to [B, T, Embed]
        ctx_emb = context.unsqueeze(1).expand(-1, T, -1)
        
        # C. Embed Trajectory
        x_emb = self.traj_emb(x_noisy) # [B, T, Embed]
        
        # D. Concatenate and Predict
        # We concatenate along the feature dimension
        h = torch.cat([x_emb, ctx_emb, t_emb], dim=-1) # [B, T, Embed*3]
        
        return self.net(h)