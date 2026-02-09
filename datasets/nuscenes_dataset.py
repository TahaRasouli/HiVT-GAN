import os
from typing import Optional, List
import torch
from torch_geometric.data import Dataset, Batch
from utils import TemporalData
from tqdm import tqdm


class NuScenesHiVTDataset(Dataset):

    # -----------------------------
    # DEFINE TRAIN CLASSES HERE
    # -----------------------------
    TRAIN_CLASSES = [0, 1, 2, 4, 5, 6]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1,
    ):
        self.split = split
        self.root = root
        self.transform = transform
        self.val_ratio = val_ratio

        self.label_map = {old: new for new, old in enumerate(self.TRAIN_CLASSES)}

        self._directory = f"{split}_processed"

        standard_path = os.path.join(self.root, self._directory)

        if os.path.isdir(standard_path):
            self._processed_dir = standard_path
            self.is_flat = False
        else:
            self._processed_dir = self.root
            self.is_flat = True

        all_files = sorted(
            f for f in os.listdir(self._processed_dir) if f.endswith(".pt")
        )

        self._processed_file_names = self._filter_files(all_files)

        if max_samples is not None:
            self._processed_file_names = self._processed_file_names[:max_samples]

        super().__init__(root, transform=transform)

    # -----------------------------
    # FILTERING (RUNS ONLY ONCE)
    # -----------------------------
    def _filter_files(self, all_files):

        print(f"[Dataset] Filtering {self.split} data...")

        limits = {0: 500, 6: 300}
        counters = {0: 0, 6: 0}

        filtered = []

        for f in tqdm(all_files, desc="Filtering Samples"):

            data = torch.load(os.path.join(self._processed_dir, f))

            m_id = self._get_maneuver_id(data)

            # ALWAYS remove invalid classes
            if m_id not in self.TRAIN_CLASSES:
                continue

            # ONLY cap training split
            if self.split == "train" and m_id in limits:
                if counters[m_id] >= limits[m_id]:
                    continue
                counters[m_id] += 1

            filtered.append(f)

        print(f"[Dataset] Filter complete. Final {self.split} set: {len(filtered)}")
        return filtered


    # -----------------------------
    # LABEL EXTRACTION
    # -----------------------------
    def _get_maneuver_id(self, data):

        mapping = {
            "follow": 0,
            "turn_left": 1,
            "turn_right": 2,
            "u_turn": 3,
            "lane_change_left": 4,
            "lane_change_right": 5,
            "stationary": 6,
            "off_map": -1
        }

        label = getattr(data, "maneuver_type", "follow")
        return mapping.get(label, 0)

    # -----------------------------
    # SANITIZE + REMAP LABEL
    # -----------------------------
    def _sanitize(self, data: TemporalData):

        m_id = self._get_maneuver_id(data)

        if m_id not in self.label_map:
            raise RuntimeError(f"Invalid maneuver id {m_id}")

        data.maneuver_id = torch.tensor(
            [self.label_map[m_id]], dtype=torch.long
        )

        data.ego_index = torch.tensor([0], dtype=torch.long)

        return data

    # -----------------------------
    # REQUIRED METHODS
    # -----------------------------
    def len(self):
        return len(self._processed_file_names)

    def get(self, idx):

        path = os.path.join(
            self._processed_dir,
            self._processed_file_names[idx]
        )

        data = torch.load(path)
        return self._sanitize(data)

    @staticmethod
    def collate_fn(batch: List[TemporalData]):
        return Batch.from_data_list(batch)
