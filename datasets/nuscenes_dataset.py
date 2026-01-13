import os
import json
import torch
from typing import List, Optional
from torch_geometric.data import Dataset, Batch
from utils import TemporalData

class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible dataset that loads specific files from a JSON split list.
    """

    def __init__(
        self,
        split_file: str,
        split: str = "train",
        tokenizer=None,
        transform=None,
        root: str = None, # Unused
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
            
        print(f"[{split.upper()}] Loaded {len(self._file_paths)} samples from split file.")

        super().__init__(root=None, transform=transform)

    def len(self) -> int:
        return len(self._file_paths)

    def get(self, idx: int) -> TemporalData:
        path = self._file_paths[idx]
        data = torch.load(path)
        data = self._sanitize(data)
        
        # Tokenization Logic
        if self.tokenizer is not None:
            caption_dict = getattr(data, 'caption_dict', {})
            raw_text = caption_dict.get('driving_behavior', "")
            
            # Encode using the tokenizer
            ids = self.tokenizer.encode(raw_text)
            
            # --- THE FIX IS HERE ---
            # Unsqueeze to [1, Seq_Len] so PyG collates to [Batch, Seq_Len]
            data.caption_ids = torch.LongTensor(ids).unsqueeze(0) 
        
        return data

    def _sanitize(self, data):
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