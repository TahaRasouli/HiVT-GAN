import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class BaseTrajectoryCritic(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        
        # 1D-CNN to process the sequence
        self.conv_block = nn.Sequential(
            spectral_norm(nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(64, 128, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(4) # Condense to 4 temporal features
        )

        dim = 128 * 4
        layers = []
        for _ in range(num_layers):
            layers.append(spectral_norm(nn.Linear(dim, hidden_dim)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.out = spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        # traj shape: [B, T, 2]
        # Calculate Derivatives (Kinematics)
        # Velocity: [B, T-1, 2]
        vel = traj[:, 1:] - traj[:, :-1]
        # Acceleration: [B, T-2, 2]
        acc = vel[:, 1:] - vel[:, :-1]
        
        # Pad to keep temporal length consistent if needed, 
        # or just use the raw derivatives. Let's use Velocity + Acc
        # [B, 4, T-2] (Concatenating Vel and Acc)
        kinematics = torch.cat([vel[:, :-1], acc], dim=-1).transpose(1, 2)
        
        features = self.conv_block(kinematics)
        flat = features.view(features.size(0), -1)
        return self.out(self.mlp(flat)).squeeze(-1)