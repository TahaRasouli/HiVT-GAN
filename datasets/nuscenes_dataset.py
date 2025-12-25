import os
from typing import Optional, List

import torch
from torch_geometric.data import Dataset, Batch

from utils import TemporalData


class NuScenesHiVTDataset(Dataset):
    """
    HiVT-compatible nuScenes dataset.

    Assumes offline preprocessing producing .pt TemporalData files:
        root/
            train_processed/
            val_processed/
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self._directory = f"{split}_processed"
        self.root = root
        self.transform = transform

        self._processed_dir = os.path.join(self.root, self._directory)
        if not os.path.isdir(self._processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {self._processed_dir}")

        self._processed_file_names = sorted(
            f for f in os.listdir(self._processed_dir) if f.endswith(".pt")
        )

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
        # Always [2, E]
        if hasattr(data, "lane_actor_index"):
            lai = data.lane_actor_index
            if not torch.is_tensor(lai):
                data.lane_actor_index = torch.empty((2, 0), dtype=torch.long)
            elif lai.numel() == 0:
                data.lane_actor_index = lai.reshape(2, 0)
            elif lai.dim() == 1 and lai.size(0) == 2:
                # occasionally stored as shape [2] (single edge) -> make [2,1]
                data.lane_actor_index = lai.reshape(2, 1)
            elif lai.dim() != 2 or lai.size(0) != 2:
                raise ValueError(f"Bad lane_actor_index shape: {tuple(lai.shape)}")

        # Always [E, 2]
        if hasattr(data, "lane_actor_vectors"):
            lav = data.lane_actor_vectors
            if not torch.is_tensor(lav):
                data.lane_actor_vectors = torch.empty((0, 2), dtype=torch.float)
            elif lav.numel() == 0:
                data.lane_actor_vectors = lav.reshape(0, 2)
            elif lav.dim() != 2 or lav.size(-1) != 2:
                raise ValueError(f"Bad lane_actor_vectors shape: {tuple(lav.shape)}")

        # Always [L, 2]
        if hasattr(data, "lane_vectors"):
            lv = data.lane_vectors
            if not torch.is_tensor(lv):
                data.lane_vectors = torch.empty((0, 2), dtype=torch.float)
            elif lv.numel() == 0:
                data.lane_vectors = lv.reshape(0, 2)
            elif lv.dim() != 2 or lv.size(-1) != 2:
                raise ValueError(f"Bad lane_vectors shape: {tuple(lv.shape)}")

        # Always [2, E]
        if hasattr(data, "edge_index"):
            ei = data.edge_index
            if ei.numel() == 0:
                data.edge_index = ei.reshape(2, 0)
            elif ei.dim() == 1 and ei.size(0) == 2:
                data.edge_index = ei.reshape(2, 1)
            elif ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError(f"Bad edge_index shape: {tuple(ei.shape)}")

        return data



    # --------------------------------------------------
    def len(self) -> int:
        return len(self._processed_file_names)

    # --------------------------------------------------
    def get(self, idx: int) -> TemporalData:
        path = os.path.join(self.processed_dir, self._processed_file_names[idx])
        data = torch.load(path)
        data = self._sanitize(data)
        assert isinstance(data, TemporalData)
        return data

    # --------------------------------------------------
    @staticmethod
    def collate_fn(batch: List[TemporalData]) -> Batch:
        return Batch.from_data_list(batch)
