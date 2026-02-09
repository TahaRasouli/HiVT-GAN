import os
from typing import Optional, List
import torch
from torch_geometric.data import Dataset, Batch
from utils import TemporalData
from tqdm import tqdm


class NuScenesHiVTDataset(Dataset):

    TRAIN_CLASSES = [0, 1, 2, 4, 5, 6]

    STR_TO_INT = {
        "follow": 0,
        "turn_left": 1,
        "turn_right": 2,
        "u_turn": 3,
        "lane_change_left": 4,
        "lane_change_right": 5,
        "stationary": 6,
        "off_map": -1,
    }

    LABEL_MAP = {old: new for new, old in enumerate(TRAIN_CLASSES)}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.val_ratio = val_ratio

        all_files = sorted(f for f in os.listdir(root) if f.endswith(".pt"))

        # ---------- REAL TRAIN / VAL SPLIT ----------
        split_index = int(len(all_files) * (1 - val_ratio))

        if split == "train":
            all_files = all_files[:split_index]
        else:
            all_files = all_files[split_index:]

        # ---------- FILTER ----------
        print(f"[Dataset] Filtering {split} data...")

        limits = {0: 500, 6: 300}
        counters = {0: 0, 6: 0}
        filtered = []

        for f in tqdm(all_files, desc="Filtering Samples"):
            data = torch.load(os.path.join(root, f))

            m_id = self._get_maneuver_id(data)

            # remove invalid labels everywhere
            if m_id not in self.TRAIN_CLASSES:
                continue

            # cap ONLY train
            if split == "train" and m_id in limits:
                if counters[m_id] >= limits[m_id]:
                    continue
                counters[m_id] += 1

            filtered.append(f)

        print(f"[Dataset] Filter complete. Final {split} set: {len(filtered)}")

        self.files = filtered

        if max_samples is not None:
            self.files = self.files[:max_samples]

        super().__init__(root, transform=transform)

    # ------------------------------------------------

    def _get_maneuver_id(self, data):
        label = getattr(data, "maneuver_type", "follow")
        return self.STR_TO_INT.get(label, -1)

    def _sanitize(self, data: TemporalData):

        old_id = self._get_maneuver_id(data)

        # safe remap
        new_id = self.LABEL_MAP[old_id]

        data.maneuver_id = torch.tensor([new_id], dtype=torch.long)
        data.ego_index = torch.tensor([0], dtype=torch.long)

        return data

    # ------------------------------------------------

    def len(self):
        return len(self.files)

    def get(self, idx):
        path = os.path.join(self.root, self.files[idx])
        data = torch.load(path)
        return self._sanitize(data)

    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)
