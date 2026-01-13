import os
from typing import Optional, List

import torch
from torch_geometric.data import Dataset, Batch

from utils import TemporalData

class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset with Captioning support.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
        tokenizer=None, # <--- NEW ARGUMENT
    ):
        self.split = split
        self._directory = f"{split}_processed"
        self.root = root
        self.transform = transform
        self.tokenizer = tokenizer # <--- STORE TOKENIZER

        self._processed_dir = os.path.join(self.root, self._directory)
        if not os.path.isdir(self._processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {self._processed_dir}")

        self._processed_file_names = sorted(
            f for f in os.listdir(self._processed_dir) if f.endswith(".pt")
        )

        if max_samples is not None:
            self._processed_file_names = self._processed_file_names[:max_samples]

        super().__init__(root, transform=transform)

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_file_names

    def _sanitize(self, data):
        # ... (Keep your existing sanitation logic here) ...
        # [Copy-paste the previous _sanitize method body here]
        
        # Always [2, E]
        if hasattr(data, "lane_actor_index"):
            lai = data.lane_actor_index
            if not torch.is_tensor(lai):
                data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
            elif lai.numel() == 0:
                data.lane_actor_index = lai.reshape(2, 0)
            elif lai.dim() == 1 and lai.size(0) == 2:
                data.lane_actor_index = lai.reshape(2, 1)

        # Always [E, 2]
        if hasattr(data, "lane_actor_vectors"):
            lav = data.lane_actor_vectors
            if not torch.is_tensor(lav):
                data.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)
            elif lav.numel() == 0:
                data.lane_actor_vectors = lav.reshape(0, 2)

        # Always [L, 2]
        if hasattr(data, "lane_vectors"):
            lv = data.lane_vectors
            if not torch.is_tensor(lv):
                data.lane_vectors = torch.empty((0, 2), dtype=torch.float)
            elif lv.numel() == 0:
                data.lane_vectors = lv.reshape(0, 2)

        # Always [2, E]
        if hasattr(data, "edge_index"):
            ei = data.edge_index
            if ei.numel() == 0:
                data.edge_index = ei.reshape(2, 0)
            elif ei.dim() == 1 and ei.size(0) == 2:
                data.edge_index = ei.reshape(2, 1)

        return data

    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int) -> TemporalData:
        path = os.path.join(self.processed_dir, self._processed_file_names[idx])
        data = torch.load(path)
        data = self._sanitize(data)
        
        # --- NEW CAPTION LOGIC ---
        if self.tokenizer is not None:
            # Extract caption or use empty string if missing
            caption_dict = getattr(data, 'caption_dict', {})
            # We strictly use 'driving_behavior' as planned
            raw_text = caption_dict.get('driving_behavior', "")
            
            # Tokenize
            ids = self.tokenizer.encode(raw_text)
            
            # Store as LongTensor
            data.caption_ids = torch.LongTensor(ids)
        
        assert isinstance(data, TemporalData)
        return data

    @staticmethod
    def collate_fn(batch: List[TemporalData]) -> Batch:
        return Batch.from_data_list(batch)