#!/usr/bin/env python3
import os
import argparse
from itertools import permutations, product
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

from utils import TemporalData


class NuScenesHiVTPreprocessor:
    def __init__(
        self,
        dataroot: str,
        version: str,
        split: str,
        out_dir: str,
        local_radius: float = 50.0,
        lane_clip_radius: float = 120.0,
        lane_resolution: float = 1.0,
    ):
        self.dataroot = dataroot
        self.version = version
        self.split = split

        # HiVT constants
        self.history_steps = 20
        self.future_steps = 30
        self.total_steps = 50

        # map parameters
        self.local_radius = local_radius              # actor–lane edges
        self.lane_clip_radius = lane_clip_radius      # lane geometry clipping
        self.lane_resolution = lane_resolution

        self.out_dir = os.path.join(out_dir, f"{split}_processed")
        os.makedirs(self.out_dir, exist_ok=True)

        print(f"[INIT] Loading NuScenes {version}")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        self.scenes = create_splits_scenes()[split]
        self.scene_map = {s["name"]: s for s in self.nusc.scene}
        self.map_cache: Dict[str, NuScenesMap] = {}

        self.sample_tokens = self._build_sample_list()
        print(f"[INIT] {len(self.sample_tokens)} samples in split '{split}'")

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    def _build_sample_list(self) -> List[str]:
        tokens = []
        for scene_name in self.scenes:
            scene = self.scene_map.get(scene_name)
            if scene is None:
                continue
            tok = scene["first_sample_token"]
            while tok:
                tokens.append(tok)
                tok = self.nusc.get("sample", tok)["next"] or None
        return tokens

    def _collect_5s_sequence(self, sample) -> List[str]:
        tokens = []
        cur = sample
        for _ in range(5):
            tokens.append(cur["token"])
            if not cur["prev"]:
                break
            cur = self.nusc.get("sample", cur["prev"])
        tokens.reverse()

        cur = sample
        for _ in range(6):
            if not cur["next"]:
                break
            cur = self.nusc.get("sample", cur["next"])
            tokens.append(cur["token"])
        return tokens

    def _interp_2hz_to_10hz(
        self, trajs_2hz: torch.Tensor, mask_2hz: torch.Tensor
        ):
        N, T_old, _ = trajs_2hz.shape
        t_old = np.linspace(0, (T_old - 1) * 0.5, T_old)
        t_new = np.linspace(0, 4.9, self.total_steps)

        trajs_10hz = torch.zeros((N, self.total_steps, 2))
        mask_10hz = torch.ones((N, self.total_steps), dtype=torch.bool)

        for i in range(N):
            valid = (~mask_2hz[i]).nonzero().squeeze(-1).numpy()
            if len(valid) < 2:
                continue

            fx = interp1d(t_old[valid], trajs_2hz[i, valid, 0].numpy(), fill_value="extrapolate")
            fy = interp1d(t_old[valid], trajs_2hz[i, valid, 1].numpy(), fill_value="extrapolate")

            trajs_10hz[i, :, 0] = torch.from_numpy(fx(t_new))
            trajs_10hz[i, :, 1] = torch.from_numpy(fy(t_new))

            t_new_t = torch.tensor(t_new, dtype=torch.float32)
            mask_10hz[i] = (t_new_t < t_old[valid[0]]) | (t_new_t > t_old[valid[-1]])

        return trajs_10hz, mask_10hz


    def _get_map(self, location: str) -> NuScenesMap:
        if location not in self.map_cache:
            self.map_cache[location] = NuScenesMap(self.dataroot, location)
        return self.map_cache[location]

    # --------------------------------------------------------------------- #
    # Lane graph (HiVT-faithful)
    # --------------------------------------------------------------------- #

    def _get_lane_features(
        self,
        nusc_map: NuScenesMap,
        origin_global: torch.Tensor,
        rot: torch.Tensor,
        node_inds: List[int],
        node_pos_t19: torch.Tensor,
    ):
        lane_pos, lane_vec, is_inter = [], [], []

        x, y = float(origin_global[0]), float(origin_global[1])
        patch = [x - 150, y - 150, x + 150, y + 150]
        records = nusc_map.get_records_in_patch(
            patch, ["lane", "lane_connector"], mode="intersect"
        )

        for layer in ["lane", "lane_connector"]:
            for token in records.get(layer, []):
                try:
                    arc = nusc_map.get_arcline_path(token)
                    poses = arcline_path_utils.discretize_lane(
                        arc, resolution_meters=self.lane_resolution
                    )
                    if not poses:
                        continue

                    pts = torch.tensor([(p[0], p[1]) for p in poses])
                    dist = torch.norm(pts - origin_global, dim=1)
                    mask = dist <= self.lane_clip_radius
                    if mask.sum() < 2:
                        continue

                    pts = torch.matmul(pts[mask] - origin_global, rot)
                    lane_pos.append(pts[:-1])
                    lane_vec.append(pts[1:] - pts[:-1])

                    is_int = int(layer == "lane_connector")                    

                    is_inter.append(
                        torch.full((pts.size(0) - 1,), is_int, dtype=torch.uint8)
                    )

                except Exception:
                    continue

        if not lane_vec:
            return (
                torch.zeros((0, 2)),
                torch.zeros(0, dtype=torch.uint8),
                torch.zeros(0, dtype=torch.uint8),
                torch.zeros(0, dtype=torch.uint8),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 2)),
            )

        lane_pos = torch.cat(lane_pos)
        lane_vec = torch.cat(lane_vec)
        is_inter = torch.cat(is_inter)

        turn = torch.zeros(len(lane_vec), dtype=torch.uint8)
        ctrl = torch.zeros(len(lane_vec), dtype=torch.uint8)

        idx = torch.LongTensor(list(product(range(len(lane_vec)), node_inds))).t()
        vec = lane_pos.repeat_interleave(len(node_inds), 0) - node_pos_t19.repeat(len(lane_vec), 1)

        keep = torch.norm(vec, dim=1) < self.local_radius
        return lane_vec, is_inter, turn, ctrl, idx[:, keep], vec[keep]

    # --------------------------------------------------------------------- #
    # Main processing
    # --------------------------------------------------------------------- #

    def process_one(self, token: str) -> Optional[TemporalData]:
        sample = self.nusc.get("sample", token)
        seq = self._collect_5s_sequence(sample)
        if len(seq) < 11:
            return None

        ego_xy = []
        ego_pose_center = None
        for i, tok in enumerate(seq):
            sd = self.nusc.get("sample_data", self.nusc.get("sample", tok)["data"]["LIDAR_TOP"])
            pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
            ego_xy.append(pose["translation"][:2])
            if i == 4:
                ego_pose_center = pose

        ego_2hz = torch.tensor(ego_xy).unsqueeze(0)
        ego_10hz, _ = self._interp_2hz_to_10hz(ego_2hz, torch.zeros((1, 11), dtype=torch.bool))
        origin = ego_10hz[0, 19]

        q = Quaternion(ego_pose_center["rotation"])
        v = q.rotate([1, 0, 0])
        theta = np.arctan2(v[1], v[0])
        rot = torch.tensor([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]],
                        dtype=torch.float32)



        actor_ids = ["ego"]
        pos = {"ego": {i: ego_xy[i] for i in range(len(seq))}}

        for t, tok in enumerate(seq):
            for ann in self.nusc.get("sample", tok)["anns"]:
                rec = self.nusc.get("sample_annotation", ann)
                pos.setdefault(rec["instance_token"], {})[t] = rec["translation"][:2]
                if rec["instance_token"] not in actor_ids:
                    actor_ids.append(rec["instance_token"])

        N = len(actor_ids)
        trajs = torch.zeros((N, len(seq), 2))
        mask = torch.ones((N, len(seq)), dtype=torch.bool)

        for i, k in enumerate(actor_ids):
            for t, xy in pos[k].items():
                trajs[i, t] = torch.tensor(xy)
                mask[i, t] = False

        trajs, mask = self._interp_2hz_to_10hz(trajs, mask)

        keep = (~mask[:, :20]).any(1)
        trajs, mask = trajs[keep], mask[keep]
        actor_ids = [a for a, k in zip(actor_ids, keep) if k]
        N = len(actor_ids)

        if actor_ids[0] != "ego":
            i = actor_ids.index("ego")
            trajs[[0, i]] = trajs[[i, 0]]
            mask[[0, i]] = mask[[i, 0]]

        pos_local = torch.matmul(trajs - origin, rot)
        padding = mask.clone()
        padding[padding[:, 19], 20:] = True
        padding[(~padding[:, :20]).sum(1) < 2, 20:] = True

        bos = torch.zeros((N, 20), dtype=torch.bool)

        bos[:, 0] = ~padding[:, 0]
        bos[:, 1:] = padding[:, :19] & ~padding[:, 1:20]


        x = pos_local.clone()

        # future (t = 20..49): relative to t=19
        x[:, 20:] = torch.where(
            (padding[:, 19].unsqueeze(-1) | padding[:, 20:]).unsqueeze(-1),
            torch.zeros_like(x[:, 20:]),
            x[:, 20:] - pos_local[:, 19].unsqueeze(1)
        )

        # history (t = 1..19): relative displacement
        x[:, 1:20] = torch.where(
            (padding[:, :19] | padding[:, 1:20]).unsqueeze(-1),
            torch.zeros_like(x[:, 1:20]),
            x[:, 1:20] - pos_local[:, :19]
        )

        # t = 0
        x[:, 0] = 0


        angles = torch.zeros(N)
        for i in range(N):
            v = pos_local[i, 19] - pos_local[i, 18]
            angles[i] = torch.atan2(v[1], v[0])

        scene = self.nusc.get("scene", sample["scene_token"])
        loc = self.nusc.get("log", scene["log_token"])["location"]
        nusc_map = self._get_map(loc)

        valid = (~padding[:, 19]).nonzero().squeeze(-1).tolist()
        lane_feats = self._get_lane_features(
            nusc_map, origin, rot, valid, pos_local[valid, 19]
        )

        return TemporalData(
            x=x[:, :20],
            y=x[:, 20:],
            positions=pos_local,
            edge_index=torch.LongTensor(list(permutations(range(N), 2))).t(),
            num_nodes=N,
            padding_mask=padding,
            bos_mask=bos,
            rotate_angles=angles,
            lane_vectors=lane_feats[0],
            is_intersections=lane_feats[1],
            turn_directions=lane_feats[2],
            traffic_controls=lane_feats[3],
            lane_actor_index=lane_feats[4],
            lane_actor_vectors=lane_feats[5],
            seq_id=int(token[:8], 16),
            av_index=0,
            agent_index=0,
            city=loc,
            origin=origin.unsqueeze(0),
            theta=torch.tensor(theta),
        )

    def process_all(self):
        for tok in tqdm(self.sample_tokens, desc=f"Processing {self.split}"):
            out = os.path.join(self.out_dir, f"{tok}.pt")
            if not os.path.exists(out):
                data = self.process_one(tok)
                if data is not None:
                    torch.save(data, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--split", choices=["train", "val"], default="train")
    args = p.parse_args()

    NuScenesHiVTPreprocessor(
        dataroot=args.dataroot,
        version=args.version,
        split=args.split,
        out_dir=args.out,
    ).process_all()


if __name__ == "__main__":
    main()
