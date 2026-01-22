import os
import json
import torch
from typing import Optional
from torch_geometric.data import Dataset, Batch
from utils import TemporalData

# --- UPDATED MAPPING ---
MANEUVER_MAP = {
    "Straight Drive": 0,
    "Left Turn": 1,
    "Right Turn": 2,
    "U-Turn": 3,
    "Lane Change Left": 4,
    "Lane Change Right": 5,
    "Stationary Stop": 6,
    "Unknown": -1
}

LANE_TYPE_MAP = {
    "Single-lane": 0, "2-lane": 1, "3-lane": 2, "4-lane": 3, "Multi-lane": 4, "Unknown": -1
}

class NuScenesHiVTDataset(Dataset):
    def __init__(
        self,
        split_file: str,
        split: str = "train",
        tokenizer=None, 
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
            # Load with weights_only=False to support custom objects
            try:
                data = torch.load(path, weights_only=False)
            except TypeError:
                data = torch.load(path)
        except Exception:
            print(f"Corrupt file: {path}")
            return self.get((idx + 1) % len(self))

        data = self._sanitize(data)
        
        # --- [FIX] INJECT MISSING ROTATE_MAT ---
        # The model expects 'rotate_mat'. If missing, we create an Identity matrix.
        if not hasattr(data, 'rotate_mat') and 'rotate_mat' not in data:
            # Determine number of nodes (agents) to shape the matrix correctly
            if hasattr(data, 'x'):
                num_nodes = data.x.size(0)
            elif hasattr(data, 'num_nodes'):
                num_nodes = data.num_nodes
            else:
                num_nodes = 1 # Fallback
            
            # Create Identity Matrix: [num_nodes, 2, 2]
            # This assumes the data is already in the correct coordinate frame 
            # or that no rotation is needed.
            identity_rot = torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(num_nodes, 1, 1)
            data.rotate_mat = identity_rot

        # --- 1. ROBUST CAPTION EXTRACTION ---
        cap_dict = {}
        if isinstance(data, dict):
             cap_dict = data.get('caption_dict', {})
        else:
             cap_dict = getattr(data, 'caption_dict', {})

        # Extract components (defaults if missing)
        man_text = cap_dict.get('maneuver_type', "")
        lane_text = cap_dict.get('lane_status', "")
        scene_desc = cap_dict.get('scene_description', "")
        
        # Fallback for category logic
        cat_str = getattr(data, 'maneuver_category', "Unknown")
        if isinstance(cat_str, list): cat_str = cat_str[0]
        
        # --- 2. CONSTRUCT FULL TEXT ---
        full_text = f"{man_text} {lane_text} {scene_desc}".strip()
        if len(full_text) < 5: full_text = "Traffic scene."

        # --- 3. BERT TOKENIZATION ---
        if self.tokenizer is not None:
            enc = self.tokenizer(
                full_text, 
                return_tensors='pt', 
                padding='max_length', 
                truncation=True, 
                max_length=64 
            )
            data.input_ids = enc['input_ids'].squeeze(0)
            data.attention_mask = enc['attention_mask'].squeeze(0)

        # --- 4. MAP LABELS FOR AUX LOSS ---
        m_id = MANEUVER_MAP.get(cat_str, -1)
        data.maneuver_id = torch.tensor([m_id], dtype=torch.long)
        data.maneuver_category = cat_str

        l_type = cap_dict.get('lane_type', "Unknown")
        l_id = -1
        for key, val in LANE_TYPE_MAP.items():
            if key in l_type: l_id = val; break
        data.lane_type_id = torch.tensor([l_id], dtype=torch.long)
        
        return data

    def _sanitize(self, data):
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