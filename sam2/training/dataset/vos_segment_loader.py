# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image as PILImage

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None
    print("Aviso: pycocotools não está instalado. Algumas funcionalidades podem não funcionar.")


class JSONSegmentLoader:
    def __init__(self, video_json_path, ann_every=1, frames_fps=24, valid_obj_ids=None):
        self.ann_every = ann_every
        self.valid_obj_ids = valid_obj_ids
        try:
            with open(video_json_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.frame_annots = data
                elif isinstance(data, dict):
                    masklet_field_name = "masklet" if "masklet" in data else "masks"
                    self.frame_annots = data[masklet_field_name]
                    if "fps" in data:
                        if isinstance(data["fps"], list):
                            annotations_fps = int(data["fps"][0])
                        else:
                            annotations_fps = int(data["fps"])
                        assert frames_fps % annotations_fps == 0
                        self.ann_every = frames_fps // annotations_fps
                else:
                    raise ValueError("Formato de dados não suportado.")
        except Exception as e:
            print(f"Erro ao carregar JSON em {video_json_path}: {e}")
            self.frame_annots = []

    def load(self, frame_id, obj_ids=None):
        try:
            assert frame_id % self.ann_every == 0
            rle_mask = self.frame_annots[frame_id // self.ann_every]

            valid_objs_ids = set(range(len(rle_mask)))
            if self.valid_obj_ids is not None:
                valid_objs_ids &= set(self.valid_obj_ids)
            if obj_ids is not None:
                valid_objs_ids &= set(obj_ids)
            valid_objs_ids = sorted(list(valid_objs_ids))

            id_2_idx = {}
            rle_mask_filtered = []
            for obj_id in valid_objs_ids:
                if rle_mask[obj_id] is not None:
                    id_2_idx[obj_id] = len(rle_mask_filtered)
                    rle_mask_filtered.append(rle_mask[obj_id])
                else:
                    id_2_idx[obj_id] = None

            raw_segments = torch.from_numpy(mask_utils.decode(rle_mask_filtered)).permute(2, 0, 1)
            segments = {}
            for obj_id in valid_objs_ids:
                if id_2_idx[obj_id] is None:
                    segments[obj_id] = None
                else:
                    idx = id_2_idx[obj_id]
                    segments[obj_id] = raw_segments[idx]
            return segments
        except Exception as e:
            print(f"Erro ao carregar segmentos no frame {frame_id}: {e}")
            return {}

    def get_valid_obj_frames_ids(self, num_frames_min=None):
        try:
            num_objects = len(self.frame_annots[0])
            res = {obj_id: [] for obj_id in range(num_objects)}

            for annot_idx, annot in enumerate(self.frame_annots):
                for obj_id in range(num_objects):
                    if annot[obj_id] is not None:
                        res[obj_id].append(int(annot_idx * self.ann_every))

            if num_frames_min is not None:
                for obj_id, valid_frames in list(res.items()):
                    if len(valid_frames) < num_frames_min:
                        res.pop(obj_id)

            return res
        except Exception as e:
            print(f"Erro ao processar quadros válidos: {e}")
            return {}


class PalettisedPNGSegmentLoader:
    def __init__(self, video_png_root):
        self.video_png_root = video_png_root
        try:
            png_filenames = os.listdir(self.video_png_root)
            self.frame_id_to_png_filename = {
                int(os.path.splitext(filename)[0]): filename for filename in png_filenames
            }
        except Exception as e:
            print(f"Erro ao inicializar PalettisedPNGSegmentLoader: {e}")
            self.frame_id_to_png_filename = {}

    def load(self, frame_id):
        try:
            mask_path = os.path.join(self.video_png_root, self.frame_id_to_png_filename[frame_id])
            masks = PILImage.open(mask_path).convert("P")
            masks = np.array(masks)
            object_id = pd.unique(masks.flatten())
            object_id = object_id[object_id != 0]
            binary_segments = {i: torch.from_numpy(masks == i) for i in object_id}
            return binary_segments
        except Exception as e:
            print(f"Erro ao carregar máscara no frame {frame_id}: {e}")
            return {}


class MultiplePNGSegmentLoader:
    def __init__(self, video_png_root, single_object_mode=False):
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode
        try:
            tmp_mask_path = (
                glob.glob(os.path.join(video_png_root, "*.png"))[0]
                if single_object_mode
                else glob.glob(os.path.join(video_png_root, "*", "*.png"))[0]
            )
            tmp_mask = np.array(PILImage.open(tmp_mask_path))
            self.H = tmp_mask.shape[0]
            self.W = tmp_mask.shape[1]
            self.obj_id = (
                int(video_png_root.split("/")[-1]) + 1 if single_object_mode else None
            )
        except Exception as e:
            print(f"Erro ao inicializar MultiplePNGSegmentLoader: {e}")
            self.H, self.W, self.obj_id = 0, 0, None

    def load(self, frame_id):
        try:
            if self.single_object_mode:
                return self._load_single_png(frame_id)
            else:
                return self._load_multiple_pngs(frame_id)
        except Exception as e:
            print(f"Erro ao carregar máscara no frame {frame_id}: {e}")
            return {}

    def _load_single_png(self, frame_id):
        try:
            mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}.png")
            mask = np.array(PILImage.open(mask_path)) if os.path.exists(mask_path) else np.zeros((self.H, self.W), dtype=bool)
            return {self.obj_id: torch.from_numpy(mask > 0)}
        except Exception as e:
            print(f"Erro ao carregar máscara única no frame {frame_id}: {e}")
            return {}

    def _load_multiple_pngs(self, frame_id):
        try:
            all_objects = sorted(glob.glob(os.path.join(self.video_png_root, "*")))
            binary_segments = {}
            for obj_folder in all_objects:
                obj_id = int(obj_folder.split("/")[-1]) + 1
                mask_path = os.path.join(obj_folder, f"{frame_id:05d}.png")
                mask = np.array(PILImage.open(mask_path)) if os.path.exists(mask_path) else np.zeros((self.H, self.W), dtype=bool)
                binary_segments[obj_id] = torch.from_numpy(mask > 0)
            return binary_segments
        except Exception as e:
            print(f"Erro ao carregar múltiplas máscaras no frame {frame_id}: {e}")
            return {}



class LazySegments:
    """
    Only decodes segments that are actually used.
    """

    def __init__(self):
        self.segments = {}
        self.cache = {}

    def __setitem__(self, key, item):
        self.segments[key] = item

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        rle = self.segments[key]
        mask = torch.from_numpy(mask_utils.decode([rle])).permute(2, 0, 1)[0]
        self.cache[key] = mask
        return mask

    def __contains__(self, key):
        return key in self.segments

    def __len__(self):
        return len(self.segments)

    def keys(self):
        return self.segments.keys()


class SA1BSegmentLoader:
    def __init__(
        self,
        video_mask_path,
        mask_area_frac_thresh=1.1,
        video_frame_path=None,
        uncertain_iou=-1,
    ):
        with open(video_mask_path, "r") as f:
            self.frame_annots = json.load(f)

        if mask_area_frac_thresh <= 1.0:
            # Lazily read frame
            orig_w, orig_h = PILImage.open(video_frame_path).size
            area = orig_w * orig_h

        self.frame_annots = self.frame_annots["annotations"]

        rle_masks = []
        for frame_annot in self.frame_annots:
            if not frame_annot["area"] > 0:
                continue
            if ("uncertain_iou" in frame_annot) and (
                frame_annot["uncertain_iou"] < uncertain_iou
            ):
                # uncertain_iou is stability score
                continue
            if (
                mask_area_frac_thresh <= 1.0
                and (frame_annot["area"] / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])

        self.segments = LazySegments()
        for i, rle in enumerate(rle_masks):
            self.segments[i] = rle

    def load(self, frame_idx):
        return self.segments
