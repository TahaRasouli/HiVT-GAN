import os
import json
import torch
from typing import List, Optional
from torch_geometric.data import Dataset, Batch
from utils import TemporalData

# --- LABEL MAPPINGS ---
# Maps VLM string outputs to Integer IDs for classification heads
MANEUVER_MAP = {
    "Straight Drive": 0, 
    "Stationary Stop": 1, 
    "Left Turn": 2, 
    "Right Turn": 3,
    "Lane Change Left": 4, 
    "Lane Change Right": 5, 
    "U-Turn": 6, 
    "Unknown": -1
}

# Simplified lane types for auxiliary task
LANE_TYPE_MAP = {
    "Single-lane": 0, 
    "2-lane": 1, 
    "3-lane": 2, 
    "4-lane": 3, 
    "Multi-lane": 4, 
    "Unknown": -1
}

class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible dataset that loads specific files from a JSON split list.
    Supports structured VLM outputs (Scene Description, Maneuver, Lane Type).
    """

    def __init__(
        self,
        split_file: str,
        split: str = "train",
        tokenizer=None,
        transform=None,
        root: str = None, # Kept for API compatibility, but unused if split_file provides full paths
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        
        # 1. Load Split File
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}. Please run the split generation script first.")
            
        with open(split_file, 'r') as f:
            splits = json.load(f)
            
        # 2. Select Split
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}. Available: {list(splits.keys())}")
            
        self._file_paths = splits[split]
        
        # 3. Optional Debug Limit
        if max_samples is not None:
            self._file_paths = self._file_paths[:max_samples]
            
        print(f"[{split.upper()}] Loaded {len(self._file_paths)} samples from {split_file}")

        super().__init__(root=None, transform=transform)

    def len(self) -> int:
        return len(self._file_paths)

    def get(self, idx: int) -> TemporalData:
        path = self._file_paths[idx]
        
        # Load with weights_only=False to support custom TemporalData objects
        try:
            data = torch.load(path, weights_only=False)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # return a dummy or handle error appropriately
            # For now, let's let it crash so you see the error, or you can implement a retry
            raise e

        data = self._sanitize(data)
        
        # --- EXTRACT VLM FIELDS ---
        # Handle cases where data might be a Dict or an Object
        if isinstance(data, dict):
             cap_dict = data.get('caption_dict', {})
             fallback_text = data.get('caption_string', "")
        else:
             cap_dict = getattr(data, 'caption_dict', {})
             fallback_text = getattr(data, 'caption_string', "")

        # 1. Text Description (for Contrastive Learning)
        raw_text = cap_dict.get('scene_description', "")
        if not raw_text:
            raw_text = fallback_text # Fallback to old data if needed

        # Tokenization
        if self.tokenizer is not None:
            # Tokenize and add batch dimension [1, Seq_Len]
            ids = self.tokenizer.encode(raw_text)
            data.caption_ids = torch.LongTensor(ids).unsqueeze(0)

        # 2. Maneuver Label (for Hard Negative Mining / Aux Loss)
        m_cat = cap_dict.get('maneuver_category', "Unknown")
        m_id = -1
        # Flexible matching (e.g., "Turn Left" matches "Left Turn")
        for key, val in MANEUVER_MAP.items():
            if key in m_cat: 
                m_id = val; break
        data.maneuver_id = torch.tensor([m_id], dtype=torch.long)

        # 3. Lane Type Label (Auxiliary Task)
        l_type = cap_dict.get('lane_type', "Unknown")
        l_id = -1
        for key, val in LANE_TYPE_MAP.items():
            if key in l_type:
                l_id = val; break
        # Fallback for complex multi-lane strings
        if l_id == -1 and "Multi" in l_type: l_id = 4 
            
        data.lane_type_id = torch.tensor([l_id], dtype=torch.long)
        
        return data

    def _sanitize(self, data):
        """
        Ensures PyG tensor dimensions are correct for batching.
        Handles empty graphs gracefully.
        """
        # 1. Check Lane Actor Index [2, E]
        if hasattr(data, "lane_actor_index"):
             lai = data.lane_actor_index
             if not torch.is_tensor(lai) or lai.numel() == 0:
                 data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
             elif lai.dim() == 1: 
                 data.lane_actor_index = lai.reshape(2, 1)

        # 2. Check Lane Actor Vectors [E, 2]
        if hasattr(data, "lane_actor_vectors"):
             lav = data.lane_actor_vectors
             if not torch.is_tensor(lav) or lav.numel() == 0:
                 data.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)

        # 3. Check Lane Vectors [L, 2]
        if hasattr(data, "lane_vectors"):
             lv = data.lane_vectors
             if not torch.is_tensor(lv) or lv.numel() == 0:
                 data.lane_vectors = torch.empty((0, 2), dtype=torch.float)
                 
        # 4. Check Edge Index [2, E]
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