import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryEncoder(nn.Module):
    """Encodes raw coordinates [30, 2] into a latent vector [128]"""
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Flatten(),
            # Assuming input is [B, 30, 2], Flatten makes it [B, 60*hidden] ? No.
            # Let's process per-step or just flatten raw input first.
            # Simpler approach for lightness: Flatten input first.
        )
        self.projection = nn.Linear(30 * 2, output_dim) 

    def forward(self, traj):
        # traj: [B, 30, 2]
        flat = traj.reshape(traj.size(0), -1) # [B, 60]
        return F.relu(self.projection(flat))

class CaptionDecoder(nn.Module):
    """GRU Decoder with Fusion"""
    def __init__(self, context_dim, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 1. Word Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Fusion Layer (Global Context + Trajectory Intent)
        self.fusion = nn.Linear(context_dim + 128, hidden_dim)
        
        # 3. GRU
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # 4. Output Head
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, global_context, traj_embedding, captions=None, teacher_forcing_ratio=0.5):
        """
        global_context: [B, 128] (From HiVT Encoder)
        traj_embedding: [B, 128] (From TrajectoryEncoder)
        captions: [B, Max_Len] (Target indices for training)
        """
        batch_size = global_context.size(0)
        
        # --- FUSION ---
        # Combine "Where I am" (Context) with "Where I'm going" (Traj)
        combined = torch.cat([global_context, traj_embedding], dim=1) # [B, 256]
        
        # Initialize GRU hidden state with this fused knowledge
        hidden = torch.tanh(self.fusion(combined)).unsqueeze(0) # [1, B, Hidden]
        
        # Start Token (Assume 1 is <SOS>)
        input_token = torch.tensor([1] * batch_size, device=global_context.device).unsqueeze(1)
        
        outputs = []
        max_len = captions.size(1) if captions is not None else 15
        
        for t in range(max_len):
            # Embed input
            embedded = self.dropout(self.embedding(input_token)) # [B, 1, Emb]
            
            # GRU Step
            output, hidden = self.gru(embedded, hidden)
            
            # Prediction
            prediction = self.fc_out(output.squeeze(1)) # [B, Vocab]
            outputs.append(prediction.unsqueeze(1))
            
            # Decide next input
            if captions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = captions[:, t].unsqueeze(1) # Teacher Forcing
            else:
                input_token = prediction.argmax(1).unsqueeze(1) # Auto-regressive
                
        return torch.cat(outputs, dim=1) # [B, Len, Vocab]

class TrajectoryCaptioner(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.traj_encoder = TrajectoryEncoder()
        self.decoder = CaptionDecoder(context_dim=128, embed_dim=128, hidden_dim=256, vocab_size=vocab_size)
        
    def forward(self, global_context, traj, captions=None):
        traj_embed = self.traj_encoder(traj)
        return self.decoder(global_context, traj_embed, captions)