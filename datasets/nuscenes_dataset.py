import os
from typing import Optional, List
import torch
from torch_geometric.data import Dataset, Batch
from utils import TemporalData

class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset.
    Modified to handle 'Flat' directory structures automatically.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1  # New arg: How much data to reserve for validation if flat
    ):
        self.split = split
        self.root = root
        self.transform = transform
        self._directory = f"{split}_processed"
        
        # 1. Check for standard folder structure (root/train_processed)
        standard_path = os.path.join(self.root, self._directory)
        
        if os.path.isdir(standard_path):
            self._processed_dir = standard_path
            self.is_flat = False
        # 2. Fallback: Check root directly (Flat structure)
        elif os.path.isdir(self.root):
            print(f"[Dataset] Warning: '{self._directory}' not found. scanning root '{self.root}' directly.")
            self._processed_dir = self.root
            self.is_flat = True
        else:
            raise FileNotFoundError(f"Could not find data in {standard_path} OR {self.root}")

        # 3. List all .pt files
        all_files = sorted(
            f for f in os.listdir(self._processed_dir) if f.endswith(".pt")
        )
        
        # 4. Handle Split Logic
        if self.is_flat:
            # If files are all in one bucket, we must split them mathematically
            # to avoid Training on Validation data.
            num_total = len(all_files)
            num_val = int(num_total * val_ratio)
            num_train = num_total - num_val
            
            if split == "train":
                self._processed_file_names = all_files[:num_train]
                print(f"[Dataset] Auto-Split: Assigned {len(self._processed_file_names)} files to TRAIN.")
            elif split == "val":
                self._processed_file_names = all_files[num_train:]
                print(f"[Dataset] Auto-Split: Assigned {len(self._processed_file_names)} files to VAL.")
            else:
                self._processed_file_names = all_files
        else:
            # Standard behavior (files are already physically separated)
            self._processed_file_names = all_files

        if max_samples is not None:
            self._processed_file_names = self._processed_file_names[:max_samples]

        super().__init__(root, transform=transform)

    # --------------------------------------------------
    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    # --------------------------------------------------
    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_file_names

    def _sanitize(self, data):
        # --- 1. Geometry Sanitization (Standard HiVT logic) ---
        if hasattr(data, "lane_actor_index"):
            lai = data.lane_actor_index
            if not torch.is_tensor(lai): data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
            elif lai.numel() == 0: data.lane_actor_index = lai.reshape(2, 0)
            elif lai.dim() == 1 and lai.size(0) == 2: data.lane_actor_index = lai.reshape(2, 1)
            elif lai.dim() != 2 or lai.size(0) != 2:
                if lai.size(1) == 2 and lai.size(0) != 2: data.lane_actor_index = lai.t()
                else: data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)

        if hasattr(data, "lane_actor_vectors"):
            lav = data.lane_actor_vectors
            if not torch.is_tensor(lav): data.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)
            elif lav.numel() == 0: data.lane_actor_vectors = lav.reshape(0, 2)
            elif lav.dim() != 2 or lav.size(-1) != 2: data.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)

        if hasattr(data, "lane_vectors"):
            lv = data.lane_vectors
            if not torch.is_tensor(lv): data.lane_vectors = torch.empty((0, 2), dtype=torch.float)
            elif lv.numel() == 0: data.lane_vectors = lv.reshape(0, 2)
            elif lv.dim() != 2 or lv.size(-1) != 2: data.lane_vectors = torch.empty((0, 2), dtype=torch.float)

        if hasattr(data, "edge_index"):
            ei = data.edge_index
            if ei.numel() == 0: data.edge_index = ei.reshape(2, 0)
            elif ei.dim() == 1 and ei.size(0) == 2: data.edge_index = ei.reshape(2, 1)
            elif ei.dim() != 2 or ei.size(0) != 2: data.edge_index = torch.empty((2, 0), dtype=torch.long)

        # --- 2. EXACT STRING MAPPING (The Fix) ---
        
        # Based on your scan results:
        str_to_int = {
            'Straight Drive': 0,
            'Left Turn': 1,
            'Right Turn': 2,
            'U-Turn': 3,
            'Lane Change Left': 4,
            'Lane Change Right': 5,
            'Stationary Stop': 6
        }

        # 1. Find the label string
        label_str = None
        
        # Check top-level attribute
        if hasattr(data, 'maneuver_category'):
            label_str = data.maneuver_category
        # Check dictionary fallback
        elif hasattr(data, 'caption_dict') and 'category' in data.caption_dict:
            label_str = data.caption_dict['category']
        # Check dictionary fallback (maneuver_category key inside dict)
        elif hasattr(data, 'caption_dict') and 'maneuver_category' in data.caption_dict:
            label_str = data.caption_dict['maneuver_category']

        # 2. Map to Integer
        if label_str in str_to_int:
            maneuver_id = str_to_int[label_str]
        else:
            # If label is missing or weird, default to Straight (0) to prevent crash
            maneuver_id = 0

        # 3. Assign as LongTensor [1]
        data.maneuver_id = torch.tensor([maneuver_id], dtype=torch.long)

        return data

    # --------------------------------------------------
    def len(self) -> int:
        return len(self._processed_file_names)

    # --------------------------------------------------
    def get(self, idx: int) -> TemporalData:
        path = os.path.join(self.processed_dir, self._processed_file_names[idx])
        data = torch.load(path)
        data = self._sanitize(data)
        return data

    # --------------------------------------------------
    @staticmethod
    def collate_fn(batch: List[TemporalData]) -> Batch:
        return Batch.from_data_list(batch)