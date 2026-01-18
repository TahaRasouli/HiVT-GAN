import os
import json
import torch
from typing import Optional
from torch_geometric.data import Dataset, Batch
from utils import TemporalData

# Mapping string labels to integers for classification heads
MANEUVER_MAP = {
    "Straight Drive": 0, "Stationary Stop": 1, "Left Turn": 2, "Right Turn": 3,
    "Lane Change Left": 4, "Lane Change Right": 5, "U-Turn": 6, "Unknown": -1
}

LANE_TYPE_MAP = {
    "Single-lane": 0, "2-lane": 1, "3-lane": 2, "4-lane": 3, "Multi-lane": 4, "Unknown": -1
}

class NuScenesHiVTDataset(Dataset):
    """
    Updated for BERT Tokenization.
    """

    def __init__(
        self,
        split_file: str,
        split: str = "train",
        tokenizer=None, # Now expects a HuggingFace Tokenizer
        transform=None,
        root: str = None, 
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            splits = json.load(f)
            
        self._file_paths = splits[split]
        if max_samples is not None:
            self._file_paths = self._file_paths[:max_samples]
            
        print(f"[{split.upper()}] Loaded {len(self._file_paths)} samples.")
        super().__init__(root=None, transform=transform)

    def len(self) -> int:
        return len(self._file_paths)

    def get(self, idx: int) -> TemporalData:
        path = self._file_paths[idx]
        try:
            data = torch.load(path, weights_only=False)
        except Exception:
            # Fallback for corrupted files
            return self.get((idx + 1) % len(self))

        data = self._sanitize(data)
        
        # --- VLM FIELDS ---
        if isinstance(data, dict):
             cap_dict = data.get('caption_dict', {})
             fallback_text = data.get('caption_string', "")
        else:
             cap_dict = getattr(data, 'caption_dict', {})
             fallback_text = getattr(data, 'caption_string', "")

        raw_text = cap_dict.get('scene_description', "")
        if not raw_text: raw_text = fallback_text
        if not raw_text: raw_text = "Traffic scene." # Safe fallback

        # --- BERT TOKENIZATION ---
        if self.tokenizer is not None:
            # Tokenize with padding/truncation
            enc = self.tokenizer(
                raw_text, 
                return_tensors='pt', 
                padding='max_length', 
                truncation=True, 
                max_length=64 # Short captions don't need 512
            )
            
            # Store inputs. We use unsqueeze(0) if the tokenizer output didn't include batch dim, 
            # but return_tensors='pt' usually gives [1, Seq]. 
            # We ensure shape is [1, Seq] so PyG collates to [Batch, Seq]
            data.input_ids = enc['input_ids'].view(1, -1)
            data.attention_mask = enc['attention_mask'].view(1, -1)

        # Labels (Same as before)
        m_cat = cap_dict.get('maneuver_category', "Unknown")
        m_id = -1
        for key, val in MANEUVER_MAP.items():
            if key in m_cat: m_id = val; break
        data.maneuver_id = torch.tensor([m_id], dtype=torch.long)

        l_type = cap_dict.get('lane_type', "Unknown")
        l_id = -1
        for key, val in LANE_TYPE_MAP.items():
            if key in l_type: l_id = val; break
        if l_id == -1 and "Multi" in l_type: l_id = 4 
        data.lane_type_id = torch.tensor([l_id], dtype=torch.long)
        
        return data

    def _sanitize(self, data):
        # ... (Keep your existing sanitize logic here, it is fine) ...
        # (Assuming you paste the previous _sanitize function here)
        if hasattr(data, "lane_actor_index"):
             lai = data.lane_actor_index
             if not torch.is_tensor(lai) or lai.numel() == 0:
                 data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
             elif lai.dim() == 1: 
                 data.lane_actor_index = lai.reshape(2, 1)

        if hasattr(data, "lane_actor_vectors"):
             lav = data.lane_actor_vectors
             if not torch.is_tensor(lav) or lav.numel() == 0:
                 data.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)

        if hasattr(data, "lane_vectors"):
             lv = data.lane_vectors
             if not torch.is_tensor(lv) or lv.numel() == 0:
                 data.lane_vectors = torch.empty((0, 2), dtype=torch.float)
                 
        if hasattr(data, "edge_index"):
             ei = data.edge_index
             if ei.numel() == 0: 
                 data.edge_index = ei.reshape(2, 0)
             elif ei.dim() == 1: 
                 data.edge_index = ei.reshape(2, 1)
        return data

    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)